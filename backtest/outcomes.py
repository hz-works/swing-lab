"""Live scorecard: what happened after each signal the scanner logged?

Reads data/signal_log.csv (written by scan.py) and measures every signal the
same way the backtest does (run.py): return from the NEXT session's open to the
close 5, 10 and 20 trading days after the signal day, minus SPY's return over
the same window ("excess return").

A stock that stays on a list for several days is counted once: a signal is
kept only if the same stock + pattern had no signal in the previous 5 sessions.

Writes
  data/signal_outcomes.csv  one row per counted signal (empty cells = too early)
  docs/outcomes.json        summary per pattern, for the page

Nothing here changes the scan itself; a failure only leaves the old files.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20)
DEDUPE_SESSIONS = 5
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def download(symbols, start):
    import yfinance as yf

    data = {}
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        raw = yf.download(chunk, start=start, auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        for s in chunk:
            try:
                df = raw[s].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df):
                data[s] = df[["Open", "Close"]]
    return data


def dedupe(log: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """First appearance only: drop a signal when the same stock + pattern was
    already logged within the previous DEDUPE_SESSIONS sessions."""
    pos = {d: i for i, d in enumerate(sessions)}
    log = log[log["date"].isin(pos)].sort_values("date")
    keep, last = [], {}
    for idx, r in log.iterrows():
        key, i = (r["symbol"], r["pattern"]), pos[r["date"]]
        if key not in last or i - last[key] > DEDUPE_SESSIONS:
            keep.append(idx)
        last[key] = i
    return log.loc[keep]


def forward(df: pd.DataFrame, day: pd.Timestamp, h: int):
    """Return from the open of the session after `day` to the close h sessions
    after `day`; None while that close does not exist yet."""
    if day not in df.index:
        return None
    i = df.index.get_loc(day)
    if i + h >= len(df):
        return None
    o = float(df["Open"].iloc[i + 1])
    c = float(df["Close"].iloc[i + h])
    if not (o > 0 and c > 0):
        return None
    return c / o - 1


def measure(log: pd.DataFrame, data: dict, spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in log.iterrows():
        out = r.to_dict()
        df = data.get(r["symbol"])
        for h in HORIZONS:
            ret = forward(df, r["date"], h) if df is not None else None
            mkt = forward(spy, r["date"], h)
            ok = ret is not None and mkt is not None
            out[f"ret_{h}d_pct"] = round(100 * ret, 2) if ok else None
            out[f"excess_{h}d_pct"] = round(100 * (ret - mkt), 2) if ok else None
        rows.append(out)
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    out = {"signals": int(len(df))}
    for h in HORIZONS:
        x = pd.to_numeric(df[f"excess_{h}d_pct"], errors="coerce").dropna()
        r = pd.to_numeric(df[f"ret_{h}d_pct"], errors="coerce").dropna()
        out[f"{h}d"] = ({"done": 0} if x.empty else {
            "done": int(len(x)),
            "avg_ret_pct": round(float(r.mean()), 2),
            "avg_excess_pct": round(float(x.mean()), 2),
            "median_excess_pct": round(float(x.median()), 2),
            "beat_market_pct": round(100 * float((x > 0).mean()), 1),
            "win_pct": round(100 * float((r > 0).mean()), 1),
        })
    return out


def summarize(res: pd.DataFrame) -> dict:
    """Per pattern: everything the scanner found vs. what the page showed.
    Halal-failed stocks are left out (they are never shown)."""
    ok = res[res["halal"].astype(str) != "fail"]
    out = {}
    for p, g in ok.groupby("pattern"):
        shown = g[g["shown"].astype(str).str.lower() == "true"]
        out[p] = {"all": stats(g), "shown": stats(shown)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="", help="offline test: folder with SYMBOL.csv price files")
    ap.add_argument("--log", default=os.path.join(ROOT, "data", "signal_log.csv"))
    ap.add_argument("--out-data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-docs", default=os.path.join(ROOT, "docs"))
    args = ap.parse_args()

    log = pd.read_csv(args.log, parse_dates=["date"])
    log = log[log["market_ok"].astype(str).str.lower() == "true"]
    symbols = sorted(log["symbol"].unique())
    start = (log["date"].min() - pd.Timedelta(days=10)).date().isoformat()
    if args.csv_dir:
        data = {f[:-4]: pd.read_csv(os.path.join(args.csv_dir, f), index_col=0, parse_dates=True)
                for f in os.listdir(args.csv_dir) if f.endswith(".csv")}
    else:
        data = download(symbols + ["SPY"], start)
    spy = data.pop("SPY")

    log = dedupe(log, spy.index)
    res = measure(log, data, spy)
    res["date"] = res["date"].dt.date.astype(str)
    os.makedirs(args.out_data, exist_ok=True)
    res.to_csv(os.path.join(args.out_data, "signal_outcomes.csv"), index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "first_signal": res["date"].min() if len(res) else None,
        "last_price_date": str(spy.index[-1].date()),
        "method": "next open -> close after 5/10/20 sessions, minus SPY; first appearance only",
        "patterns": summarize(res) if len(res) else {},
    }
    os.makedirs(args.out_docs, exist_ok=True)
    with open(os.path.join(args.out_docs, "outcomes.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"signals counted: {len(res)} (from {summary['first_signal']})")
    for p, s in summary["patterns"].items():
        a = s["all"]
        line = " · ".join(
            f"{h}d: {a[f'{h}d']['avg_excess_pct']}% ({a[f'{h}d']['done']})" if a[f"{h}d"]["done"]
            else f"{h}d: -" for h in HORIZONS)
        print(f"{p:9s} {a['signals']:4d} signals | excess {line}")


if __name__ == "__main__":
    main()
