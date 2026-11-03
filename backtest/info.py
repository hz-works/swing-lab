"""Extra info for the stocks shown: next earnings date and a short news brief."""
from datetime import date, datetime, timedelta, timezone


def _parse_time(v):
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _news_item(a: dict):
    c = a.get("content") or a
    title = c.get("title")
    if not title:
        return None
    url = ((c.get("clickThroughUrl") or {}).get("url") or (c.get("canonicalUrl") or {}).get("url")
           or c.get("link") or "")
    provider = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
    when = _parse_time(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"))
    return {"title": title, "url": url, "provider": provider,
            "time_utc": when.strftime("%Y-%m-%d %H:%M") if when else None, "_when": when}


def fetch_extras(symbol: str, max_news: int = 2, news_days: int = 3) -> dict:
    import yfinance as yf

    t = yf.Ticker(symbol)
    out = {"earnings_date": None, "earnings_in_days": None, "news": []}
    today = date.today()
    try:
        cal = t.calendar or {}
        dates = [d for d in cal.get("Earnings Date", []) if d and d >= today]
        if dates:
            nxt = min(dates)
            out["earnings_date"] = nxt.isoformat()
            out["earnings_in_days"] = (nxt - today).days
    except Exception as exc:
        print(f"calendar failed for {symbol}: {exc}")
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=news_days)
        items = [_news_item(a) for a in (t.get_news(count=10) or [])]
        items = [i for i in items if i and (i["_when"] is None or i["_when"] >= cutoff)]
        items.sort(key=lambda i: i["_when"] or cutoff, reverse=True)
        out["news"] = [{k: v for k, v in i.items() if k != "_when"} for i in items[:max_news]]
    except Exception as exc:
        print(f"news failed for {symbol}: {exc}")
    return out
