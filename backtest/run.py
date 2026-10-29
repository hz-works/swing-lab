"""Backtest the four swing patterns and write results/ (summary + trades)."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from patterns import indicators, simulate
from universe import load_universe

NAMES_AR = {
    "pullback": "ارتداد الترند",
    "breakout": "الاختراق",
    "ep": "فجوة الأرباح (EP)",
    "rsi2": "التشبع البيعي RSI-2",
}


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


def max_drawdown_r(r_by_exit: pd.Series) -> float:
    eq = r_by_exit.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def summarize(tr: pd.DataFrame) -> dict:
    r = tr["R"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = -losses.sum()
    years = (tr["exit_date"].max() - tr["entry_date"].min()).days / 365.25
    by_year = tr.groupby(tr["exit_date"].dt.year)["R"].agg(["count", "mean"]).round(3)
    return {
        "trades": int(len(r)),
        "trades_per_year": round(len(r) / years, 1) if years > 0 else None,
        "win_rate_pct": round(100 * len(wins) / len(r), 1),
        "avg_win_R": round(float(wins.mean()), 2) if len(wins) else 0,
        "avg_loss_R": round(float(losses.mean()), 2) if len(losses) else 0,
        "expectancy_R": round(float(r.mean()), 3),
        "profit_factor": round(float(wins.sum() / gross_loss), 2) if gross_loss > 0 else None,
        "max_drawdown_R": round(max_drawdown_r(tr.sort_values("exit_date")["R"]), 1),
        "avg_days_held": round(float(tr["days"].mean()), 1),
        "by_year": {int(y): {"trades": int(row["count"]), "expectancy_R": float(row["mean"])}
                    for y, row in by_year.iterrows()},
    }


def write_markdown(summary: dict, meta: dict, path: str):
    lines = [
        "# نتائج الاختبار التاريخي — أنماط السوينج",
        "",
        f"الفترة: {meta['start']} حتى {meta['end']} · الأسهم المختبرة: {meta['symbols']}",
        "",
        "| النمط | الصفقات | بالسنة | نسبة الربح | متوسط الربح | متوسط الخسارة | التوقع لكل صفقة | معامل الربح | أسوأ تراجع |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p, s in summary.items():
        lines.append(
            f"| {NAMES_AR[p]} | {s['trades']} | {s['trades_per_year']} | {s['win_rate_pct']}% "
            f"| {s['avg_win_R']}R | {s['avg_loss_R']}R | {s['expectancy_R']}R "
            f"| {s['profit_factor']} | {s['max_drawdown_R']}R |"
        )
    lines += ["", "R = مقدار المخاطرة في الصفقة (المسافة بين الدخول والوقف).", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--data-start", default="2014-06-01")
    ap.add_argument("--limit", type=int, default=0, help="test with first N symbols")
    ap.add_argument("--csv-dir", default="", help="read OHLCV CSVs instead of downloading")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.csv_dir:
        data = {f[:-4]: pd.read_csv(os.path.join(args.csv_dir, f), index_col=0, parse_dates=True)
                for f in os.listdir(args.csv_dir) if f.endswith(".csv")}
    else:
        symbols = load_universe()["Symbol"].tolist()
        if args.limit:
            symbols = symbols[: args.limit]
        data = download(symbols, args.data_start)

    trades = []
    for sym, df in data.items():
        d = indicators(df)
        for t in simulate(d, args.start):
            t["symbol"] = sym
            trades.append(t)
    tr = pd.DataFrame(trades)
    os.makedirs(args.out, exist_ok=True)
    if tr.empty:
        print("no trades")
        return
    tr = tr.drop(columns=["entry_i", "exit_i"])
    tr["R"] = tr["R"].round(3)
    tr.to_csv(os.path.join(args.out, "trades.csv"), index=False)

    summary = {p: summarize(tr[tr["pattern"] == p]) for p in NAMES_AR if (tr["pattern"] == p).any()}
    meta = {"start": args.start, "end": str(tr["exit_date"].max().date()), "symbols": len(data),
            "slippage_per_side_pct": 0.05}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "patterns": summary}, fh, ensure_ascii=False, indent=2, default=str)
    write_markdown(summary, meta, os.path.join(args.out, "summary.md"))
    print(open(os.path.join(args.out, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
