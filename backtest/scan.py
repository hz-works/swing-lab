"""Daily swing scanner: lists today's stocks per pattern. No entries/stops -
entry, stop and size are decided by hand.

Shown (ranked, top 5 each), only when SPY is above its 200-day average:
  1. EP (earnings/news gap) - gap in the last 5 days, still above gap-day low
  2. RSI-2 oversold in an uptrend
  3. Trend pullback + stock in the top 20 % by 6-month return
Logged only (not shown): breakout.

Every candidate goes through the AAOIFI halal screen (halal.py); non-compliant
stocks are dropped from the lists (still logged).

Writes docs/signals.json, docs/latest.md and appends data/signal_log.csv.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from halal import screen
from patterns import indicators, sig_breakout, sig_ep, sig_pullback, sig_rsi2
from universe import load_universe

TOP_N = 5
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = {
    "ep": "فجوة الأرباح",
    "rsi2": "التشبع البيعي RSI-2",
    "pullback": "ارتداد الترند",
    "breakout": "الاختراق (سجل فقط)",
}


def download(symbols):
    import yfinance as yf

    data = {}
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        raw = yf.download(chunk, period="2y", auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        for s in chunk:
            try:
                df = raw[s].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) > 260:
                data[s] = df[["Open", "High", "Low", "Close", "Volume"]]
        print(f"downloaded {min(k + 100, len(symbols))}/{len(symbols)} -> {len(data)}")
    return data


def row(sym, d, sector, extra):
    last = d.iloc[-1]
    prev = d["Close"].iloc[-2]
    r = {
        "symbol": sym,
        "sector": sector,
        "close": round(float(last["Close"]), 2),
        "change_pct": round(100 * (float(last["Close"]) / float(prev) - 1), 2),
        "rvol": round(float(last["Volume"] / last["vol50"]), 2) if last["vol50"] else None,
        "adr_pct": round(float(last["adr"]), 2),
        "rsi2": round(float(last["rsi2"]), 1),
        "tradingview": f"https://www.tradingview.com/chart/?symbol={sym.replace('-', '.')}",
    }
    r.update(extra)
    return r


def scan(data, sectors, spy):
    closes = pd.DataFrame({s: df["Close"] for s, df in data.items()})
    ret126 = closes.pct_change(126, fill_method=None).iloc[-1]
    rs_pct = ret126.rank(pct=True)
    spy_c = spy["Close"]
    market_ok = bool(spy_c.iloc[-1] > spy_c.rolling(200).mean().iloc[-1])
    asof = spy.index[-1]

    found = {p: [] for p in NAMES}
    for s, df in data.items():
        if df.index[-1] != asof:  # stale / halted
            continue
        d = indicators(df)
        rs = float(rs_pct.get(s, np.nan))
        rs_r = None if np.isnan(rs) else round(100 * rs)

        ep = sig_ep(d).to_numpy()
        n = len(d)
        for g in range(n - 1, max(n - 6, 0), -1):
            if ep[g]:
                if (d["Close"].iloc[g + 1:] >= d["Low"].iloc[g]).all():
                    gd = d.iloc[g]
                    found["ep"].append(row(s, d, sectors.get(s, ""), {
                        "days_since_gap": n - 1 - g,
                        "gap_pct": round(100 * (gd["Close"] / d["Close"].iloc[g - 1] - 1), 1),
                        "gap_vol_x": round(float(gd["Volume"] / d["vol50"].iloc[g - 1]), 1),
                        "gap_day_high": round(float(gd["High"]), 2),
                        "gap_day_low": round(float(gd["Low"]), 2),
                        "rs_pct": rs_r, "_score": float(gd["Volume"] / d["vol50"].iloc[g - 1]),
                    }))
                break
        if bool(sig_rsi2(d).iloc[-1]):
            found["rsi2"].append(row(s, d, sectors.get(s, ""), {
                "rs_pct": rs_r, "_score": -float(d["rsi2"].iloc[-1])}))
        if bool(sig_pullback(d).iloc[-1]) and rs_r is not None and rs_r >= 80:
            found["pullback"].append(row(s, d, sectors.get(s, ""), {
                "rs_pct": rs_r, "_score": rs}))
        if bool(sig_breakout(d).iloc[-1]):
            found["breakout"].append(row(s, d, sectors.get(s, ""), {
                "rs_pct": rs_r, "_score": rs if not np.isnan(rs) else 0}))

    for p in found:
        found[p].sort(key=lambda r: r["_score"], reverse=True)
    return asof, market_ok, found


def write_outputs(asof, market_ok, found, spy):
    date = str(asof.date())
    spy_c = spy["Close"]
    ok = {p: [r for r in rows if r["halal_status"] != "fail"] for p, rows in found.items()}
    shown = {p: (ok[p][:TOP_N] if market_ok else []) for p in ("ep", "rsi2", "pullback")}
    out = {
        "date": date,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "market": {"ok": market_ok, "spy": round(float(spy_c.iloc[-1]), 2),
                   "spy_sma200": round(float(spy_c.rolling(200).mean().iloc[-1]), 2)},
        "counts": {p: len(v) for p, v in ok.items()},
        "halal_excluded": {p: len(found[p]) - len(ok[p]) for p in found},
        "patterns": {p: [{k: v for k, v in r.items() if k != "_score"} for r in rows]
                     for p, rows in shown.items()},
    }
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "signals.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    L = [f"# ماسح السوينج — {date}", ""]
    if market_ok:
        L.append(f"✅ السوق إيجابي: SPY {out['market']['spy']} فوق متوسط 200 ({out['market']['spy_sma200']})")
    else:
        L.append(f"⛔ السوق ضعيف: SPY {out['market']['spy']} تحت متوسط 200 ({out['market']['spy_sma200']}) — لا إشارات اليوم")
    L.append("")
    for p, rows in out["patterns"].items():
        L += [f"## {NAMES[p]} ({out['counts'][p]} إشارة حلال، أفضل {len(rows)} · "
              f"استُبعد {out['halal_excluded'][p]} غير متوافق)", ""]
        if not rows:
            L += ["لا يوجد", ""]
            continue
        L += ["| السهم | السعر | التغير | الحجم النسبي | التذبذب | RSI2 | قوة السهم | الحلال | ملاحظة |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            note = ""
            if p == "ep":
                note = f"فجوة {r['gap_pct']}% قبل {r['days_since_gap']} يوم، حجم ×{r['gap_vol_x']}، قمة {r['gap_day_high']}"
            L.append(f"| [{r['symbol']}]({r['tradingview']}) | {r['close']} | {r['change_pct']}% | "
                     f"×{r['rvol']} | {r['adr_pct']}% | {r['rsi2']} | {r['rs_pct']} | {r['halal']} | {note} |")
        L.append("")
    L.append("الحلال: فحص النشاط ونسبتي الديون والنقد (AAOIFI). نسبة الدخل المحرّم غير متوفرة مجاناً — أكّدها في Zoya قبل الدخول.")
    with open(os.path.join(ROOT, "docs", "latest.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    # signal log: every signal (shown or not) for later evaluation
    log_rows = []
    for p, rows in found.items():
        shown_syms = {r["symbol"] for r in shown.get(p, [])}
        for rank, r in enumerate(rows, 1):
            log_rows.append({"date": date, "symbol": r["symbol"], "pattern": p, "rank": rank,
                             "close": r["close"], "market_ok": market_ok,
                             "halal": r["halal_status"], "shown": r["symbol"] in shown_syms})
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    path = os.path.join(ROOT, "data", "signal_log.csv")
    log = pd.DataFrame(log_rows)
    if os.path.exists(path):
        log = pd.concat([pd.read_csv(path), log], ignore_index=True)
    if not log.empty:
        log = log.drop_duplicates(["date", "symbol", "pattern"], keep="last")
    log.to_csv(path, index=False)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="")
    args = ap.parse_args()
    if args.csv_dir:
        data = {f[:-4]: pd.read_csv(os.path.join(args.csv_dir, f), index_col=0, parse_dates=True)
                for f in os.listdir(args.csv_dir) if f.endswith(".csv")}
        sectors = {}
    else:
        uni = load_universe()
        sectors = dict(zip(uni["Symbol"], uni["Sector"]))
        data = download(uni["Symbol"].tolist() + ["SPY"])
    spy = data.pop("SPY")
    asof, market_ok, found = scan(data, sectors, spy)
    candidates = sorted({r["symbol"] for rows in found.values() for r in rows})
    if args.csv_dir:  # offline test: no fundamentals
        verdicts = {s: {"status": "doubtful", "label": "⚠️ اختبار", "debt_pct": None, "cash_pct": None}
                    for s in candidates}
    else:
        verdicts = screen(candidates, os.path.join(ROOT, "data", "halal_cache.json"))
    for rows in found.values():
        for r in rows:
            v = verdicts[r["symbol"]]
            r.update(halal=v["label"], halal_status=v["status"],
                     debt_pct=v["debt_pct"], cash_pct=v["cash_pct"])
    write_outputs(asof, market_ok, found, spy)


if __name__ == "__main__":
    main()
