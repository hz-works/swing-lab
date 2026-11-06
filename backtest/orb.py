"""Gap-day opening-range breakout (ORB) backtest - the plan the earnings radar shows.

Question: when a liquid stock gaps up >= 4 % at the open, does buying the break of
the first 15-minute high (stop under the 15-minute low) make money?

Data
  * daily bars (yfinance, unadjusted) -> find the gap days
  * 5-minute bars (Alpaca, full SIP feed, free plan: data older than 15 minutes)
    -> simulate the day. Needs the secrets ALPACA_KEY_ID / ALPACA_SECRET_KEY.

Rules (decided BEFORE looking at results, not tuned)
  * universe: S&P 1500 after the sector/business halal pre-filter (universe.py)
  * gap day: open >= +4 % over the previous close, previous close >= $10,
    20-day avg volume >= 1M shares and >= $20M traded per day
  * opening range (OR) = 09:30-09:45; skip if the OR closes below the open
    (the gap is already fading)
  * entry: first 5-minute bar that trades above the OR high, until 11:30;
    fill at max(bar open, OR high) + 0.1 % slippage
  * stop: OR low - 0.1 % slippage. A stop inside the entry bar counts as hit.
  * exits compared: (a) close of the day, (b) +2R target else close,
    (c) hold 5 days: stop stays, exit at the close of day 5
  * results in R (multiples of the risk), with and without the SPY > 200-day filter

Limits: gap days are a proxy for earnings/news (no historical earnings calendar);
today's index members only (survivorship bias); no halal debt ratios in history.

Writes results/orb_summary.md and results/orb_trades.csv.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

from universe import load_universe

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
NY = "America/New_York"
SLIP = 0.001
GAP_MIN = 0.04
LAST_ENTRY = "11:30"
HOLD_DAYS = 5


# ------------------------------------------------------------------ data
def daily(symbols, start):
    import yfinance as yf

    out = {}
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        raw = yf.download(chunk, start=start, auto_adjust=False, group_by="ticker",
                          threads=True, progress=False)
        for s in chunk:
            try:
                df = raw[s].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) > 60:
                out[s] = df[["Open", "High", "Low", "Close", "Volume"]]
        print(f"daily {min(k + 100, len(symbols))}/{len(symbols)} -> {len(out)}")
    return out


def gap_events(data, spy, start):
    spy_ok = (spy["Close"] > spy["Close"].rolling(200).mean()).shift(1).fillna(False).astype(bool)
    rows = []
    for s, d in data.items():
        prev = d["Close"].shift(1)
        vol20 = d["Volume"].rolling(20).mean().shift(1)
        dvol20 = (d["Close"] * d["Volume"]).rolling(20).mean().shift(1)
        ok = ((d["Open"] >= prev * (1 + GAP_MIN)) & (prev >= 10)
              & (vol20 >= 1_000_000) & (dvol20 >= 20_000_000))
        ok &= d.index >= pd.Timestamp(start)
        for i in np.flatnonzero(ok.to_numpy()):
            day = d.index[i]
            nxt = d.iloc[i + 1: i + 1 + HOLD_DAYS]
            rows.append({"date": day.date().isoformat(), "symbol": s,
                         "prev_close": float(prev.iloc[i]), "vol20": float(vol20.iloc[i]),
                         "market_ok": bool(spy_ok.get(day, False)),
                         "next_lows": nxt["Low"].tolist(), "next_closes": nxt["Close"].tolist()})
    ev = pd.DataFrame(rows)
    print(f"gap events: {len(ev)} on {ev['date'].nunique() if len(ev) else 0} days")
    return ev


class Alpaca:
    def __init__(self):
        self.h = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
                  "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
        self.last = 0.0

    def _get(self, params):
        for attempt in range(5):
            wait = 0.32 - (time.time() - self.last)  # stay under 200 calls/min
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            r = requests.get(BARS_URL, headers=self.h, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def bars_5m(self, day, symbols):
        """symbol -> DataFrame of regular-session 5-minute bars (NY time index)."""
        alp = {s.replace("-", "."): s for s in symbols}
        params = {"symbols": ",".join(alp), "timeframe": "5Min", "feed": "sip",
                  "adjustment": "raw", "limit": 10000,
                  "start": f"{day}T13:00:00Z", "end": f"{day}T21:05:00Z"}
        got = {}
        while True:
            js = self._get(params)
            for sym, bars in (js.get("bars") or {}).items():
                got.setdefault(sym, []).extend(bars)
            tok = js.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok
        out = {}
        for sym, bars in got.items():
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
            df = df.set_index("t").sort_index()
            hm = df.index.strftime("%H:%M")
            df = df[(hm >= "09:30") & (hm < "16:00")]
            if len(df):
                out[alp.get(sym, sym)] = df[["o", "h", "l", "c", "v"]]
        return out


# ------------------------------------------------------------ simulation
def simulate(b: pd.DataFrame, ev: dict):
    hm = b.index.strftime("%H:%M")
    orb = b[hm < "09:45"]
    if len(orb) < 2 or hm[0] != "09:30":
        return None
    open_px = float(b["o"].iloc[0])
    gap = open_px / ev["prev_close"] - 1
    if gap < GAP_MIN:
        return None
    or_hi, or_lo = float(orb["h"].max()), float(orb["l"].min())
    if float(orb["c"].iloc[-1]) < open_px:
        return None  # opening range faded below the open
    rvol15 = float(orb["v"].sum()) / (ev["vol20"] / 26)

    after = b[(hm >= "09:45")]
    k = next((i for i, (t, r) in enumerate(after.iterrows())
              if t.strftime("%H:%M") <= LAST_ENTRY and r["h"] > or_hi), None)
    if k is None:
        return None
    bar = after.iloc[k]
    entry = max(float(bar["o"]), or_hi) * (1 + SLIP)
    stop = or_lo * (1 - SLIP)
    risk = entry - stop
    if risk <= 0:
        return None
    tgt = entry + 2 * risk

    rest = after.iloc[k:]
    eod = float(rest["c"].iloc[-1]) * (1 - SLIP)
    r_eod = r_2r = None
    stopped = False
    for j, (_, x) in enumerate(rest.iterrows()):
        if x["l"] <= stop:  # includes the entry bar (conservative)
            px = (stop if j == 0 else min(float(x["o"]), stop)) * (1 - SLIP)
            r_eod = (px - entry) / risk
            r_2r = r_2r if r_2r is not None else r_eod
            stopped = True
            break
        if r_2r is None and x["h"] >= tgt:
            r_2r = (max(float(x["o"]), tgt) * (1 - SLIP) - entry) / risk if j else 2.0
    if not stopped:
        r_eod = (eod - entry) / risk
        r_2r = r_2r if r_2r is not None else r_eod

    # (c) hold 5 days with the same stop
    if stopped:
        r_5d = r_eod
    else:
        r_5d = None
        for lo, cl in zip(ev["next_lows"], ev["next_closes"]):
            if lo <= stop:
                r_5d = (stop * (1 - SLIP) - entry) / risk
                break
        if r_5d is None:
            last = ev["next_closes"][-1] if ev["next_closes"] else eod
            r_5d = (last * (1 - SLIP) - entry) / risk

    return {"date": ev["date"], "symbol": ev["symbol"], "gap_pct": round(100 * gap, 2),
            "rvol15": round(rvol15, 2), "market_ok": ev["market_ok"],
            "entry": round(entry, 4), "stop": round(stop, 4),
            "risk_pct": round(100 * risk / entry, 2), "entry_time": after.index[k].strftime("%H:%M"),
            "R_eod": round(r_eod, 3), "R_2r": round(r_2r, 3), "R_5d": round(r_5d, 3)}


# --------------------------------------------------------------- report
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"n": 0}
    win, loss = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win": round(100 * (r > 0).mean(), 1), "avg": round(r.mean(), 3),
            "pf": round(win / loss, 2) if loss else None, "total": round(r.sum(), 1)}


GROUPS = [
    ("كل الفجوات ≥ 4%", lambda t: t),
    ("فجوة ≥ 8%", lambda t: t[t.gap_pct >= 8]),
    ("حجم أول 15 دقيقة ×3+", lambda t: t[t.rvol15 >= 3]),
    ("حجم أول 15 دقيقة ×6+", lambda t: t[t.rvol15 >= 6]),
    ("فجوة ≥ 8% + حجم ×3+", lambda t: t[(t.gap_pct >= 8) & (t.rvol15 >= 3)]),
]


def report(t: pd.DataFrame, ev_n: int, out_dir: str):
    L = ["# اختبار خطة يوم الفجوة (ORB)", "",
         f"الفترة: {t['date'].min()} حتى {t['date'].max()} · أيام فجوة ≥4%: {ev_n} · صفقات دخلت: {len(t)}",
         "", "R = مضاعف المخاطرة (1R = المسافة بين الدخول والوقف). متوسط موجب = ربح على المدى الطويل.", ""]
    for title, mkt in (("مع فلتر السوق (SPY فوق 200)", True), ("بدون فلتر السوق", False)):
        base = t[t.market_ok] if mkt else t
        L += [f"## {title}", "",
              "| المجموعة | الخروج | صفقات | ربح % | متوسط R | معامل الربح | مجموع R |",
              "|---|---|---|---|---|---|---|"]
        for name, f in GROUPS:
            sub = f(base)
            for col, lbl in (("R_eod", "إغلاق اليوم"), ("R_2r", "هدف 2R"), ("R_5d", "احتفاظ 5 أيام")):
                s = stats(sub[col])
                if s["n"]:
                    L.append(f"| {name} | {lbl} | {s['n']} | {s['win']}% | {s['avg']} | {s['pf']} | {s['total']} |")
        L.append("")
    best = t[t.market_ok]
    L += ["## حسب السنة (مع فلتر السوق، خروج إغلاق اليوم)", "",
          "| السنة | صفقات | ربح % | متوسط R |", "|---|---|---|---|"]
    for y, g in best.groupby(best["date"].str[:4]):
        s = stats(g["R_eod"])
        L.append(f"| {y} | {s['n']} | {s['win']}% | {s['avg']} |")
    L += ["", "حدود الاختبار: الفجوات تقريب لأيام الأرباح/الأخبار، أعضاء المؤشر الحاليين فقط، "
              "انزلاق 0.1% لكل جهة، وقف داخل شمعة الدخول يُحسب ضرباً."]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "orb_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    t.to_csv(os.path.join(out_dir, "orb_trades.csv"), index=False)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--max-days", type=int, default=0, help="limit gap days (quick test)")
    args = ap.parse_args()

    uni = load_universe()
    hist_start = (pd.Timestamp(args.start) - pd.Timedelta(days=420)).date().isoformat()
    data = daily(uni["Symbol"].tolist() + ["SPY"], hist_start)
    spy = data.pop("SPY")
    ev = gap_events(data, spy, args.start)
    days = sorted(ev["date"].unique())
    if args.max_days:
        days = days[-args.max_days:]

    api, trades = Alpaca(), []
    for n, day in enumerate(days, 1):
        todays = ev[ev["date"] == day]
        try:
            bars = api.bars_5m(day, todays["symbol"].tolist())
        except Exception as exc:
            print(f"{day}: bars failed ({exc})"[:200])
            continue
        for e in todays.to_dict("records"):
            if e["symbol"] in bars:
                r = simulate(bars[e["symbol"]], e)
                if r:
                    trades.append(r)
        if n % 100 == 0:
            print(f"{n}/{len(days)} days -> {len(trades)} trades")
    report(pd.DataFrame(trades), len(ev), args.out)


if __name__ == "__main__":
    main()
