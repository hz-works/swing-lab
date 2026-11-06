"""Earnings radar: which stocks can gap at the NEXT session because of earnings.

Gap candidates for the next session N (today = T, the last daily bar):
  * reported on T after the close (AMC)  -> the after-hours move is already known
  * report on N before the open (BMO)
  * report on N with unknown timing      -> shown, flagged as "timing unknown"

Source: Yahoo earnings calendar (yfinance); Nasdaq's public calendar API as
fallback. Both are free and may change without notice, so every call is guarded:
a failure just gives an empty radar.
"""
import time
from datetime import date, timedelta

import pandas as pd
import requests

NY = "America/New_York"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}
MIN_MCAP = 500_000_000


def next_session(today: date) -> date:
    d = today + timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun (exchange holidays are not handled)
        d += timedelta(days=1)
    return d


def _num(v):
    try:
        x = float(str(v).replace("$", "").replace(",", "").replace("(", "-").replace(")", ""))
        return None if pd.isna(x) else x
    except (TypeError, ValueError):
        return None


def _timing(code: str, ts: pd.Timestamp) -> str:
    """Yahoo often says 'TAS' (time as scheduled) - then the event time decides."""
    if code in ("BMO", "AMC"):
        return code
    hm = ts.hour * 60 + ts.minute
    if hm == 0:
        return "TNS"
    if hm < 9 * 60 + 30:
        return "BMO"
    if hm >= 16 * 60:
        return "AMC"
    return "DMH"  # during market hours


def _yahoo(start: date, end: date) -> list:
    import yfinance as yf

    cal = yf.Calendars()
    rows, offset = [], 0
    while offset < 1500:
        for attempt in range(3):  # Yahoo sometimes answers 429 / "Invalid Crumb" once
            try:
                df = cal.get_earnings_calendar(market_cap=MIN_MCAP, filter_most_active=False,
                                               start=start.isoformat(),
                                               end=(end + timedelta(days=1)).isoformat(),
                                               limit=100, offset=offset, force=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
        if df is None or df.empty:
            break
        for sym, r in df.iterrows():
            ts = pd.Timestamp(r.get("Event Start Date"))
            ts = (ts.tz_localize("UTC") if ts.tzinfo is None else ts).tz_convert(NY)
            rows.append({"symbol": str(sym).replace(".", "-"),
                         "company": r.get("Company") or "",
                         "date": ts.date(),
                         "timing": _timing(str(r.get("Timing") or "").upper(), ts),
                         "eps_est": _num(r.get("EPS Estimate")),
                         "eps_act": _num(r.get("Reported EPS")),
                         "surprise_pct": _num(r.get("Surprise(%)"))})
        if len(df) < 100:
            break
        offset += 100
    return rows


NASDAQ_TIMING = {"time-pre-market": "BMO", "time-after-hours": "AMC"}


def _nasdaq(start: date, end: date) -> list:
    rows, d = [], start
    while d <= end:
        r = requests.get("https://api.nasdaq.com/api/calendar/earnings",
                         params={"date": d.isoformat()}, headers=UA, timeout=20)
        for x in ((r.json().get("data") or {}).get("rows") or []):
            rows.append({"symbol": str(x.get("symbol", "")).replace(".", "-"),
                         "company": x.get("name") or "",
                         "date": d,
                         "timing": NASDAQ_TIMING.get(x.get("time"), "TNS"),
                         "eps_est": _num(x.get("epsForecast")),
                         "eps_act": None, "surprise_pct": None})
        d += timedelta(days=1)
    return rows


def reporters(today: date) -> dict:
    """symbol -> earnings info for stocks that can gap at the next session."""
    nxt = next_session(today)
    rows, src = [], None
    for name, fn in (("yahoo", _yahoo), ("nasdaq", _nasdaq)):
        try:
            rows = fn(today, nxt)
        except Exception as exc:
            print(f"earnings calendar ({name}) failed: {type(exc).__name__}: {exc}"[:200])
            rows = []
        if rows:
            src = name
            break
    out = {}
    for r in rows:
        if r["date"] == today and r["timing"] == "AMC":
            r["when"] = "after_close_today"
        elif r["date"] == nxt and r["timing"] == "BMO":
            r["when"] = "before_open"
        elif r["date"] == nxt and r["timing"] == "TNS":
            r["when"] = "unknown"
        else:
            continue
        r["date"] = r["date"].isoformat()
        out.setdefault(r["symbol"], r)
    print(f"earnings radar: {len(out)} reporters for {nxt} (source: {src})")
    return out


def after_hours(symbols) -> dict:
    """symbol -> post-market change % (only meaningful after the close)."""
    import yfinance as yf

    out = {}
    for s in symbols:
        try:
            info = yf.Ticker(s).info or {}
            post, reg = info.get("postMarketPrice"), info.get("regularMarketPrice")
            if post and reg:
                out[s] = round(100 * (float(post) / float(reg) - 1), 2)
        except Exception as exc:
            print(f"after-hours failed for {s}: {exc}"[:160])
    return out
