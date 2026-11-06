"""Daily swing scanner: lists today's stocks per pattern plus the ready-made
order for the next session. Position size stays a manual decision.

Shown (ranked, top 5 each), only when SPY is above its 200-day average:
  1. EP (earnings/news gap) - gap in the last 5 days, still above gap-day low
  2. RSI-2 oversold in an uptrend
  3. Trend pullback + stock in the top 20 % by 6-month return
Logged only (not shown): breakout.

Every candidate goes through the AAOIFI halal screen (halal.py); non-compliant
stocks are dropped from the lists (still logged).

Tomorrow's orders: for every listed stock the exact pending order for the next
session, built with the same entry/stop rules as the backtest (patterns.py):
  * EP       - Buy Stop at the gap-day high (only while the high is still
               unbroken and within 5 days of the gap), stop = today's low
  * pullback - Buy Stop above today's high, stop = lowest low of last 5 days
  * RSI-2    - buy at tomorrow's open, stop = today's low, exit on the first
               close above the 5-day average (max 10 days)
Orders whose stop is wider than 1 ADR are dropped (the backtest skips them).

Earnings radar (radar.py): liquid halal stocks that report after today's close
or before tomorrow's open - the gap-day itself is where EP pays most. Each gets
a watch-only note: buying the gap day itself (opening-range breakout) showed no
edge in results/orb_summary.md (4,613 trades, profit factor 0.86); the entry
comes later from tomorrow's orders (break of the gap-day high).

Watch list: strong stocks in a clean uptrend drifting toward their 10/20-day
average but not there yet - the next pullback signals.

Writes docs/signals.json, docs/latest.md and appends data/signal_log.csv.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from halal import screen
from info import fetch_extras
from quality import SCORERS
from radar import after_hours, reporters
from patterns import indicators, sig_breakout, sig_ep, sig_pullback, sig_rsi2
from universe import load_universe

TOP_N = 5
RADAR_N = 10
WATCH_N = 5
GAP_MIN = 0.04  # radar plan: gap of at least +4 %
ENRICH_N = 10  # candidates per pattern that get earnings/news + quality score
ORDERS_PER_PATTERN = 3
LIMIT_CUSHION = 0.015  # stop-limit: limit = trigger + 1.5 %
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = {
    "ep": "فجوة الأرباح",
    "rsi2": "التشبع البيعي RSI-2",
    "pullback": "ارتداد الترند",
    "breakout": "الاختراق (سجل فقط)",
    "radar": "رادار الأرباح",
    "watch": "قريبة من الإشارة",
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


def make_order(kind, entry, stop, adr_pct):
    """Pending order for the next session; None when the stop is invalid or
    wider than 1 ADR (same rule as the backtest)."""
    entry, stop = float(entry), float(stop)
    risk = entry - stop
    if risk <= 0 or risk / entry < 0.005 or risk / entry * 100 > float(adr_pct):
        return None
    o = {
        "type": kind,  # "buy_stop" | "open"
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "risk_pct": round(100 * risk / entry, 2),
        "target_2r": round(entry + 2 * risk, 2),
        "shares_per_100": int(100 // risk),
    }
    if kind == "buy_stop":
        o["limit"] = round(entry * (1 + LIMIT_CUSHION), 2)
    return o


def liquid(d) -> bool:
    last = d.iloc[-1]
    return bool(last["Close"] >= 10 and last["vol20"] >= 1_000_000 and last["dvol20"] >= 20_000_000)


def uptrend(d) -> bool:
    c = d["Close"]
    t = ((c > d["sma50"]) & (d["sma50"] > d["sma150"]) & (d["sma150"] > d["sma200"])
         & (d["sma200"] > d["sma200"].shift(20)))
    return bool(t.iloc[-1])


def radar_row(s, d, sector, rep, rs_r):
    last = d.iloc[-1]
    close = float(last["Close"])
    hi = float(last["hi252"])
    return row(s, d, sector, {
        "company": rep.get("company", ""),
        "when": rep["when"],
        "report_date": rep["date"],
        "eps_est": rep.get("eps_est"), "eps_act": rep.get("eps_act"),
        "surprise_pct": rep.get("surprise_pct"),
        "gap_trigger": round(close * (1 + GAP_MIN), 2),
        "uptrend": uptrend(d),
        "dist_high_pct": round(100 * (close / hi - 1), 1) if hi else None,
        "rs_pct": rs_r, "_score": rs_r or 0,
    })


def watch_row(s, d, sector, rs_r):
    """Strong uptrend, pulled back from the high, approaching EMA10/20 from above."""
    last = d.iloc[-1]
    if not (uptrend(d) and liquid(d) and 2 <= last["adr"] <= 6):
        return None
    if last["Close"] > 0.97 * d["High"].iloc[-10:].max():
        return None  # not pulled back yet
    for ma in ("ema10", "ema20"):  # the first average below price is the next target
        lvl = float(last[ma])
        dist = 100 * (float(last["Low"]) / lvl - 1)
        if dist <= 2:
            return None  # already touching -> pullback territory, not "approaching"
        if dist <= 5:
            return row(s, d, sector, {"near_ma": ma, "ma_level": round(lvl, 2),
                                      "dist_ma_pct": round(dist, 1), "rs_pct": rs_r,
                                      "_score": rs_r - dist})
    return None


def scan(data, sectors, spy, reps=None):
    reps = reps or {}
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
                    unbroken = not (d["High"].iloc[g + 1:] > gd["High"]).any()
                    ep_order = (make_order("buy_stop", gd["High"], d["Low"].iloc[-1],
                                           d["adr"].iloc[g - 1])
                                if unbroken and n - 1 - g <= 4 else None)
                    found["ep"].append(row(s, d, sectors.get(s, ""), {
                        "days_since_gap": n - 1 - g,
                        "gap_pct": round(100 * (gd["Close"] / d["Close"].iloc[g - 1] - 1), 1),
                        "gap_vol_x": round(float(gd["Volume"] / d["vol50"].iloc[g - 1]), 1),
                        "gap_day_high": round(float(gd["High"]), 2),
                        "gap_day_low": round(float(gd["Low"]), 2),
                        "pre_run_pct": round(100 * (float(d["Close"].iloc[g - 1]) /
                                                    float(d["Close"].iloc[max(0, g - 61)]) - 1), 1),
                        "rs_pct": rs_r, "_score": float(gd["Volume"] / d["vol50"].iloc[g - 1]),
                        "order": ep_order,
                    }))
                break
        if bool(sig_rsi2(d).iloc[-1]):
            last = d.iloc[-1]
            found["rsi2"].append(row(s, d, sectors.get(s, ""), {
                "order": make_order("open", last["Close"], last["Low"], last["adr"]),
                "rs_pct": rs_r, "_score": -float(d["rsi2"].iloc[-1]),
                "dist200_pct": round(100 * (float(d["Close"].iloc[-1]) / float(d["sma200"].iloc[-1]) - 1), 1)}))
        if bool(sig_pullback(d).iloc[-1]) and rs_r is not None and rs_r >= 80:
            last = d.iloc[-1]
            near = next((ma for ma in ("ema10", "ema20", "sma50")
                         if last["Low"] <= last[ma] * 1.02 and last["Close"] >= last[ma] * 0.98), None)
            found["pullback"].append(row(s, d, sectors.get(s, ""), {
                "rs_pct": rs_r, "_score": rs, "near_ma": near,
                "order": make_order("buy_stop", last["High"], d["Low"].iloc[-5:].min(), last["adr"])}))
        if bool(sig_breakout(d).iloc[-1]):
            found["breakout"].append(row(s, d, sectors.get(s, ""), {
                "rs_pct": rs_r, "_score": rs if not np.isnan(rs) else 0}))
        if s in reps and liquid(d):
            found["radar"].append(radar_row(s, d, sectors.get(s, ""), reps[s], rs_r))
        if rs_r is not None and rs_r >= 80 and not bool(sig_pullback(d).iloc[-1]):
            w = watch_row(s, d, sectors.get(s, ""), rs_r)
            if w:
                found["watch"].append(w)

    for p in found:
        found[p].sort(key=lambda r: r["_score"], reverse=True)
    return asof, market_ok, found


def strip(rows):
    return [{k: v for k, v in r.items() if k != "_score"} for r in rows]


def build_orders(ok):
    """Best few halal stocks per pattern that have a valid order for tomorrow."""
    out = []
    for p in ("ep", "pullback", "rsi2"):
        rows = [r for r in ok[p] if r.get("order")][:ORDERS_PER_PATTERN]
        out += [dict({k: v for k, v in r.items() if k != "_score"}, pattern=p) for r in rows]
    return out


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
        "orders": build_orders(ok) if market_ok else [],
        "radar": strip(ok["radar"][:RADAR_N]) if market_ok else [],
        "watch": strip(ok["watch"][:WATCH_N]) if market_ok else [],
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
    WHEN = {"after_close_today": "أعلنت اليوم بعد الإغلاق", "before_open": "تعلن قبل الافتتاح",
            "unknown": "تعلن الجلسة القادمة (التوقيت غير معروف)"}
    L += [f"## 📅 رادار الأرباح ({len(out['radar'])})", ""]
    if out["radar"]:
        L += ["| السهم | التوقيت | بعد الإغلاق | المفاجأة | سعر +4% | ترند | قوة | الحلال |",
              "|---|---|---|---|---|---|---|---|"]
        for r in out["radar"]:
            ah = f"{r['ah_pct']}%" if r.get("ah_pct") is not None else "–"
            sp = f"{r['surprise_pct']}%" if r.get("surprise_pct") is not None else "–"
            L.append(f"| [{r['symbol']}]({r['tradingview']}) | {WHEN[r['when']]} | {ah} | {sp} | "
                     f"{r['gap_trigger']} | {'✅' if r['uptrend'] else '–'} | {r['rs_pct']} | {r['halal']} |")
        L += ["", "👀 للمراقبة فقط — لا شراء يوم الفجوة (اختبار 4,613 صفقة: معامل ربح 0.86). "
                  "إذا عمل فجوة وسكّر قوي، بيطلع بأوامر الغد بأمر Buy Stop فوق قمة يوم الفجوة.", ""]
    else:
        L += ["لا يوجد", ""]
    L += [f"## 🎯 قريبة من الإشارة ({len(out['watch'])})", ""]
    for r in out["watch"]:
        L.append(f"- [{r['symbol']}]({r['tradingview']}) — يبعد {r['dist_ma_pct']}% عن "
                 f"{r['near_ma'].upper()} ({r['ma_level']}) · قوة {r['rs_pct']} · {r['halal']}")
    L.append("")
    L += [f"## 📋 أوامر الغد ({len(out['orders'])})", ""]
    if out["orders"]:
        L += ["| السهم | النمط | الأمر | الدخول | الوقف | المخاطرة | هدف 2R | أسهم لكل $100 مخاطرة |",
              "|---|---|---|---|---|---|---|---|"]
        for r in out["orders"]:
            o = r["order"]
            kind = (f"Buy Stop-Limit {o['entry']} / {o['limit']}" if o["type"] == "buy_stop"
                    else "شراء عند الافتتاح")
            L.append(f"| [{r['symbol']}]({r['tradingview']}) | {NAMES[r['pattern']]} | {kind} | "
                     f"{o['entry']} | {o['stop']} | {o['risk_pct']}% | {o['target_2r']} | {o['shares_per_100']} |")
    else:
        L.append("لا أوامر للغد")
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
        sectors, names = {}, {}
    else:
        uni = load_universe()
        sectors = dict(zip(uni["Symbol"], uni["Sector"]))
        names = dict(zip(uni["Symbol"], uni["Name"]))
        data = download(uni["Symbol"].tolist() + ["SPY"])
    spy = data.pop("SPY")
    reps = {} if args.csv_dir else reporters(spy.index[-1].date())
    asof, market_ok, found = scan(data, sectors, spy, reps)
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
    if not args.csv_dir:
        amc = [r["symbol"] for r in found["radar"]
               if r["halal_status"] != "fail" and r["when"] == "after_close_today"][:25]
        ah = after_hours(amc)
        for r in found["radar"]:
            r["ah_pct"] = ah.get(r["symbol"])
    enrich(found, names, offline=bool(args.csv_dir))
    write_outputs(asof, market_ok, found, spy)


def enrich(found, names, offline=False):
    """Earnings date, news and quality stars for the best halal candidates,
    then re-rank each pattern by stars (ties keep the pattern's own order)."""
    cache = {}
    for p, scorer in SCORERS.items():
        ok = [r for r in found[p] if r["halal_status"] != "fail"]
        top, rest = ok[:ENRICH_N], ok[ENRICH_N:]
        for r in top:
            if r["symbol"] not in cache:
                cache[r["symbol"]] = ({"earnings_date": None, "earnings_in_days": None, "news": []}
                                      if offline else fetch_extras(r["symbol"], names.get(r["symbol"], "")))
            r.update(cache[r["symbol"]])
            r["stars"], r["reasons"] = scorer(r)
        if p == "radar":  # a confirmed after-hours jump beats everything else
            top.sort(key=lambda r: (-int((r.get("ah_pct") or 0) >= 4), -r["stars"]))
        else:
            top.sort(key=lambda r: -r["stars"])  # stable sort keeps original order on ties
        failed = [r for r in found[p] if r["halal_status"] == "fail"]
        found[p] = top + rest + failed


if __name__ == "__main__":
    main()
