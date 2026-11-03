"""Extra info for the stocks shown: next earnings date and a short news brief.

News: Yahoo (yfinance) first; if it returns nothing, Google News RSS as fallback.
"""
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests

HEADERS = {"User-Agent": "swing-lab/1.0 (https://github.com/hz-works/swing-lab)"}


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
    return {"title": title, "url": url, "provider": provider, "_when": when}


SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "holdings", "ltd",
            "ltd.", "plc", "group", "the", "class", "a", "b", "n.v.", "s.a.", "lp", "l.p."}


def company_core(company: str) -> str:
    words = [w for w in company.replace(",", " ").split() if w.lower() not in SUFFIXES]
    return " ".join(words) if words else company


COMMON = {"best", "flex", "target", "apple", "match", "block", "visa", "general", "first", "united",
          "american", "national", "international", "global", "advanced", "applied", "digital", "energy",
          "health", "home", "royal", "southern", "western", "north", "south", "central", "public",
          "regional", "standard", "smart", "power", "pacific", "atlantic", "universal", "graphic"}


def is_relevant(title: str, symbol: str, company: str) -> bool:
    """Keep a headline only if it names the company (distinctive first word, e.g.
    'Glaukos', 'Palantir') or shows the ticker in ticker form ('(DT)', 'NYSE:DT', '$DT').
    Plain words like 'path', 'wing', 'flex' or 'best' do not count."""
    t = title.lower()
    key = company_core(company).split()[0].lower() if company else ""
    if key and len(key) >= 5 and key not in COMMON and re.search(rf"\b{re.escape(key)}", t):
        return True
    s = re.escape(symbol.lower())
    return bool(re.search(rf"(\({s}\)|:{s}\b|\${s}\b)", t))


def _yahoo_news(t):
    return [i for i in (_news_item(a) for a in (t.get_news(count=10) or [])) if i]


def _google_news(symbol: str, company: str = ""):
    q = (f'"{company_core(company)}" stock' if company else f'"{symbol}" stock') + " when:3d"
    url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
    root = ET.fromstring(requests.get(url, headers=HEADERS, timeout=20).content)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        try:
            when = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            when = None
        if title:
            items.append({"title": title, "url": it.findtext("link") or "", "provider": source, "_when": when})
    return items


def translate_ar(texts):
    """Machine-translate short headlines to Arabic (free Google Translate endpoint).
    Returns a list of the same length; None where translation failed."""
    out = []
    for text in texts:
        try:
            r = requests.get("https://translate.googleapis.com/translate_a/single",
                             params={"client": "gtx", "sl": "en", "tl": "ar", "dt": "t", "q": text},
                             headers=HEADERS, timeout=15)
            out.append("".join(seg[0] for seg in r.json()[0] if seg and seg[0]) or None)
        except Exception as exc:
            print(f"translate failed: {exc}")
            out.append(None)
    return out


def fetch_extras(symbol: str, company: str = "", max_news: int = 2, news_days: int = 3) -> dict:
    import yfinance as yf

    t = yf.Ticker(symbol)
    out = {"earnings_date": None, "earnings_in_days": None, "news": [], "news_src": None}
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

    cutoff = datetime.now(timezone.utc) - timedelta(days=news_days)
    items, errors = [], []
    for name, getter in (("yahoo", lambda: _yahoo_news(t)), ("google", lambda: _google_news(symbol, company))):
        try:
            items = [i for i in getter() if (i["_when"] is None or i["_when"] >= cutoff)
                     and (not company or is_relevant(i["title"], symbol, company))]
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:160])
            items = []
        if items:
            out["news_src"] = name
            break
    if errors:
        out["news_err"] = errors
        print(f"news errors for {symbol}: {errors}")
    items.sort(key=lambda i: i["_when"] or cutoff, reverse=True)
    out["news"] = [{"title": i["title"], "url": i["url"], "provider": i["provider"],
                    "time_utc": i["_when"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M") if i["_when"] else None}
                   for i in items[:max_news]]
    for n, ar in zip(out["news"], translate_ar([n["title"] for n in out["news"]])):
        n["title_ar"] = ar
    return out
