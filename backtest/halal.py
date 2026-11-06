"""Sharia screen (AAOIFI standard, as used by Zoya) for scanner candidates.

  1. Business activity: forbidden industries fail, doubtful ones are flagged.
  2. Interest-bearing debt / 36-month average market cap  <= 30 %
  3. Cash + interest-bearing securities / 36-month average market cap <= 30 %
     (AAOIFI and Zoya use the trailing 36-month average, not today's market
     cap - a stock that doubled this year can pass on today's value and still
     fail the standard, e.g. MPC in Nov 2026)
  4. Non-permissible income <= 5 % of revenue  -> NOT available for free,
     so every pass still says "confirm in Zoya".

Only the stocks the scanner found are screened (not the whole universe);
results are cached for 30 days in data/halal_cache.json.
"""
import json
import os
import time
from datetime import date

FORBIDDEN = (
    "bank", "insurance", "credit services", "capital markets", "mortgage",
    "asset management", "financial conglomerate", "brewer", "winer", "distiller",
    "tobacco", "gambling", "casino", "adult",
)
DOUBTFUL = (
    "restaurant", "lodging", "travel services", "entertainment",
    "aerospace & defense", "reit", "real estate",
)
MAX_AGE_DAYS = 30


def avg_mcap(info: dict) -> float:
    """36-month average market cap = average close of 36 months x shares;
    falls back to today's market cap when the history is missing."""
    shares, avg_close = info.get("sharesOutstanding"), info.get("avgClose36m")
    if shares and avg_close:
        return float(shares) * float(avg_close)
    return float(info.get("marketCap") or 0)


def classify(info: dict) -> dict:
    industry = (info.get("industry") or "").lower()
    sector = (info.get("sector") or "").lower()
    mcap = avg_mcap(info)
    debt = info.get("totalDebt") or 0
    cash = info.get("totalCash") or 0
    out = {"industry": info.get("industry") or "", "debt_pct": None, "cash_pct": None}

    if sector == "financial services" or any(k in industry for k in FORBIDDEN):
        return {**out, "status": "fail", "label": f"❌ نشاط غير متوافق ({out['industry'] or sector})"}
    if not mcap:
        return {**out, "status": "doubtful", "label": "⚠️ لا توجد بيانات مالية — تحقق في Zoya"}

    out["debt_pct"] = round(100 * debt / mcap, 1)
    out["cash_pct"] = round(100 * cash / mcap, 1)
    ratios = f"الديون {out['debt_pct']}%، النقد {out['cash_pct']}%"
    if out["debt_pct"] > 30 or out["cash_pct"] > 30:
        return {**out, "status": "fail", "label": f"❌ نسب مالية فوق 30% ({ratios})"}
    hit = next((k for k in DOUBTFUL if k in industry or k in sector), None)
    if hit:
        return {**out, "status": "doubtful",
                "label": f"⚠️ نشاط مشكوك ({out['industry']}) — {ratios} — تحقق في Zoya"}
    return {**out, "status": "pass", "label": f"✅ متوافق مبدئياً ({ratios}) — أكّد الدخل المحرّم في Zoya"}


def _fetch(symbol: str) -> dict:
    import yfinance as yf

    t = yf.Ticker(symbol)
    info = t.info or {}
    out = {k: info.get(k) for k in ("industry", "sector", "marketCap", "totalDebt", "totalCash",
                                    "sharesOutstanding")}
    try:
        hist = t.history(period="3y", auto_adjust=False)["Close"].dropna()
        out["avgClose36m"] = round(float(hist.mean()), 4) if len(hist) > 200 else None
    except Exception as exc:
        print(f"history failed for {symbol}: {exc}")
        out["avgClose36m"] = None
    return out


def screen(symbols, cache_path: str, fetch=_fetch) -> dict:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    today = date.today()
    verdicts = {}
    for s in symbols:
        entry = cache.get(s)
        fresh = (entry and "avgClose36m" in entry.get("info", {})  # old entries lack the 36m average
                 and (today - date.fromisoformat(entry["checked"])).days <= MAX_AGE_DAYS)
        if not fresh:
            try:
                info = fetch(s)
                entry = {"checked": today.isoformat(), "info": info}
                cache[s] = entry
            except Exception as exc:  # keep going; unknown -> doubtful
                print(f"halal fetch failed for {s}: {exc}")
                entry = {"checked": today.isoformat(), "info": {}}
            time.sleep(0.3)
        verdicts[s] = classify(entry["info"])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return verdicts
