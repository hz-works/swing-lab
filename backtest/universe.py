"""Build the test universe: S&P 500 + 400 + 600 (large, mid and small caps).

Approximate halal pre-filter (sector/business only, no debt ratios yet):
excludes Financials and sub-industries for alcohol, tobacco and gambling.
"""
import io

import pandas as pd
import requests

HEADERS = {"User-Agent": "swing-lab/1.0 (https://github.com/hz-works/swing-lab)"}
PAGES = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
FALLBACK_SP500 = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
EXCLUDED_SECTORS = {"Financials"}
EXCLUDED_SUB_INDUSTRIES = {
    "Brewers",
    "Distillers & Vintners",
    "Tobacco",
    "Casinos & Gaming",
}


def _find_table(html: str) -> pd.DataFrame:
    for table in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in table.columns]
        sym = next((c for c in cols if c.lower() in ("symbol", "ticker", "ticker symbol")), None)
        if sym and any("sector" in c.lower() for c in cols):
            table = table.rename(columns={sym: "Symbol"})
            sector = next(c for c in cols if "sector" in c.lower())
            sub = next((c for c in cols if "sub-industry" in c.lower()), None)
            out = pd.DataFrame(
                {
                    "Symbol": table["Symbol"].astype(str),
                    "Sector": table[sector].astype(str),
                    "SubIndustry": table[sub].astype(str) if sub else "",
                }
            )
            return out
    raise ValueError("constituents table not found")


def load_universe() -> pd.DataFrame:
    frames = []
    for name, url in PAGES.items():
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
            df = _find_table(html)
            df["Index"] = name
            frames.append(df)
            print(f"{name}: {len(df)} tickers")
        except Exception as exc:  # keep going with what we have
            print(f"{name}: failed ({exc})")
    if not frames:
        df = pd.read_csv(FALLBACK_SP500).rename(columns={"GICS Sector": "Sector"})
        df["SubIndustry"] = df.get("GICS Sub-Industry", "")
        df["Index"] = "SP500"
        frames.append(df[["Symbol", "Sector", "SubIndustry", "Index"]])
    uni = pd.concat(frames, ignore_index=True).drop_duplicates("Symbol")
    uni["Symbol"] = uni["Symbol"].str.strip().str.replace(".", "-", regex=False)
    before = len(uni)
    uni = uni[~uni["Sector"].isin(EXCLUDED_SECTORS)]
    uni = uni[~uni["SubIndustry"].isin(EXCLUDED_SUB_INDUSTRIES)]
    print(f"universe: {before} -> {len(uni)} after sector/business halal pre-filter")
    return uni.reset_index(drop=True)
