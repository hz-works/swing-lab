"""Pattern-quality test: after a pattern appears, does the stock beat the market?

No entries, stops or position sizes are simulated (those are decided by hand).
For every signal on day t we measure the return from the NEXT day's open to the
close 5, 10 and 20 trading days later, minus SPY's return over the same window
("excess return"). Each pattern is compared with a baseline: all liquid
stock-days in the universe, so we see whether the pattern adds anything.

Variants:
  all      every signal
  market   only when SPY closes above its 200-day average
  market+RS  market filter + stock in the top 20 % by 6-month return
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from patterns import base_filter, indicators, sig_breakout, sig_ep, sig_pullback, sig_rsi2
from universe import load_universe

HORIZONS = (5, 10, 20)
PATTERNS = {
    "pullback": ("ارتداد الترند", sig_pullback, True),
    "breakout": ("الاختراق", sig_breakout, True),
    "ep": ("فجوة الأرباح (EP)", sig_ep, False),
    "rsi2": ("التشبع البيعي RSI-2", sig_rsi2, False),
}
VARIANT_AR = {"all": "كل الإشارات", "market": "مع فلتر السوق", "market+RS": "فلتر السوق + قوة السهم"}


def download(symbols, start):
    import yfinance as yf

    data = {}
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, auto_adjust=True, group_by="ticker",
                                  threads=True, progress=False)
                break
            except Exception as exc:
                print(f"download retry {attempt + 1}: {exc}")
                time.sleep(20)
        else:
            continue
        for s in chunk:
            try:
                df = raw[s].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) > 300:
                data[s] = df[["Open", "High", "Low", "Close", "Volume"]]
        print(f"downloaded {min(k + 100, len(symbols))}/{len(symbols)} -> {len(data)} usable")
        time.sleep(2)
    return data


def dedupe(sig: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """Keep a signal only if the same stock had no signal in the previous `days` days."""
    prev = sig.shift(1, fill_value=False).astype(int).rolling(days, min_periods=1).max()
    return sig & ~prev.astype(bool)


def stats(mask: pd.DataFrame, excess: dict, raw20: pd.DataFrame, years: float) -> dict:
    m = mask.to_numpy()
    out = {"signals": int(m.sum())}
    out["per_year"] = round(out["signals"] / years, 1)
    if out["signals"] == 0:
        return out
    for h in HORIZONS:
        x = excess[h].to_numpy()[m]
        x = x[~np.isnan(x)]
        out[f"excess_{h}d_pct"] = round(100 * float(np.mean(x)), 2)
    x20 = excess[20].to_numpy()[m]
    x20 = x20[~np.isnan(x20)]
    out["median_excess_20d_pct"] = round(100 * float(np.median(x20)), 2)
    out["beat_market_20d_pct"] = round(100 * float(np.mean(x20 > 0)), 1)
    r20 = raw20.to_numpy()[m]
    out["raw_20d_pct"] = round(100 * float(np.nanmean(r20)), 2)
    by_year = {}
    dates = np.repeat(mask.index.year.to_numpy()[:, None], mask.shape[1], axis=1)[m]
    x10 = excess[10].to_numpy()[m]
    for y in np.unique(dates):
        sel = (dates == y) & ~np.isnan(x10)
        if sel.any():
            by_year[int(y)] = {"signals": int(sel.sum()),
                               "excess_10d_pct": round(100 * float(np.mean(x10[sel])), 2)}
    out["by_year"] = by_year
    return out


def write_markdown(res: dict, meta: dict, path: str):
    L = [
        "# جودة الأنماط — هل يتفوق السهم على السوق بعد ظهور النمط؟",
        "",
        f"الفترة: {meta['start']} حتى {meta['end']} · الأسهم: {meta['symbols']} · "
        "الدخول افتراضياً عند افتتاح اليوم التالي · بدون تكاليف",
        "",
        "العائد الزائد = عائد السهم ناقص عائد SPY في نفس الفترة.",
        "",
        "| النمط | النسخة | إشارات بالسنة | زائد 5 أيام | زائد 10 أيام | زائد 20 يوم | تفوّق على السوق (20 يوم) |",
        "|---|---|---|---|---|---|---|",
    ]
    b = res["baseline"]
    L.append(f"| **المرجع: كل الأسهم** | أيام عشوائية | — | {b['excess_5d_pct']}% | "
             f"{b['excess_10d_pct']}% | {b['excess_20d_pct']}% | {b['beat_market_20d_pct']}% |")
    for p, variants in res["patterns"].items():
        for v, s in variants.items():
            if s["signals"] == 0:
                continue
            L.append(f"| {PATTERNS[p][0]} | {VARIANT_AR[v]} | {s['per_year']} | {s['excess_5d_pct']}% "
                     f"| {s['excess_10d_pct']}% | {s['excess_20d_pct']}% | {s['beat_market_20d_pct']}% |")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--data-start", default="2014-06-01")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv-dir", default="")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.csv_dir:
        data = {f[:-4]: pd.read_csv(os.path.join(args.csv_dir, f), index_col=0, parse_dates=True)
                for f in os.listdir(args.csv_dir) if f.endswith(".csv")}
    else:
        symbols = load_universe()["Symbol"].tolist()
        if args.limit:
            symbols = symbols[: args.limit]
        data = download(symbols + ["SPY"], args.data_start)
    spy = data.pop("SPY")
    idx = spy.index

    opens = pd.DataFrame({s: df["Open"] for s, df in data.items()}).reindex(idx)
    closes = pd.DataFrame({s: df["Close"] for s, df in data.items()}).reindex(idx)
    spy_c, spy_o = spy["Close"], spy["Open"]
    market_ok = (spy_c > spy_c.rolling(200).mean()).to_numpy()[:, None]
    rs_top = (closes.pct_change(126, fill_method=None).rank(axis=1, pct=True) >= 0.8).to_numpy()

    excess, raw20 = {}, None
    for h in HORIZONS:
        stock = closes.shift(-h) / opens.shift(-1) - 1
        bench = spy_c.shift(-h) / spy_o.shift(-1) - 1
        excess[h] = stock.sub(bench, axis=0)
        if h == 20:
            raw20 = stock

    sigs = {p: {} for p in PATTERNS}
    base = {}
    for s, df in data.items():
        d = indicators(df)
        base[s] = base_filter(d, 8).reindex(idx, fill_value=False)
        for p, (_, fn, _) in PATTERNS.items():
            sigs[p][s] = fn(d).fillna(False).astype(bool).reindex(idx, fill_value=False)

    in_period = np.asarray(idx >= pd.Timestamp(args.start))[:, None]
    has_fwd = excess[20].notna().to_numpy()
    valid = in_period & has_fwd
    years = (idx[-21] - pd.Timestamp(args.start)).days / 365.25

    base_mask = pd.DataFrame(base).reindex(columns=closes.columns).to_numpy() & valid
    res = {"baseline": stats(pd.DataFrame(base_mask, index=idx), excess, raw20, years), "patterns": {}}
    for p, (_, _, use_rs) in PATTERNS.items():
        sig = dedupe(pd.DataFrame(sigs[p]).reindex(columns=closes.columns)).to_numpy() & valid
        variants = {"all": sig, "market": sig & market_ok}
        if use_rs:
            variants["market+RS"] = sig & market_ok & rs_top
        res["patterns"][p] = {v: stats(pd.DataFrame(m, index=idx), excess, raw20, years)
                              for v, m in variants.items()}

    os.makedirs(args.out, exist_ok=True)
    for old in ("trades.csv",):
        if os.path.exists(os.path.join(args.out, old)):
            os.remove(os.path.join(args.out, old))
    meta = {"start": args.start, "end": str(idx[-21].date()), "symbols": len(data),
            "entry": "next open", "costs": "none"}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, **res}, fh, ensure_ascii=False, indent=2)
    write_markdown(res, meta, os.path.join(args.out, "summary.md"))
    print(open(os.path.join(args.out, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
