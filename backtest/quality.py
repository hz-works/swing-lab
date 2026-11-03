"""Quality score (1-5 stars) per pattern, from the published rules of each setup.

Each rule adds or removes a point; the reasons are returned as short keys that
the page translates (Arabic / English).
"""


def _clamp(points: int) -> int:
    return max(1, min(5, points))


def score_ep(r: dict) -> tuple:
    pts, why = 1, []
    vx = r.get("gap_vol_x") or 0
    if vx >= 5:
        pts += 1; why.append("vol5")
    if vx >= 10:
        pts += 1; why.append("vol10")
    if r.get("days_since_gap", 9) <= 2:
        pts += 1; why.append("fresh")
    if r.get("close", 0) >= r.get("gap_day_high", 1e18):
        pts += 1; why.append("above_high")
    if r.get("pre_run_pct") is not None and r["pre_run_pct"] < 20:
        pts += 1; why.append("neglected")
    return _clamp(pts), why


def score_rsi2(r: dict) -> tuple:
    pts, why = 1, []
    rs = r.get("rs_pct") or 0
    if rs >= 80:
        pts += 1; why.append("rs80")
    if rs >= 95:
        pts += 1; why.append("rs95")
    if (r.get("rsi2") or 99) < 5:
        pts += 1; why.append("rsi5")
    if (r.get("change_pct") or 0) > -7:
        pts += 1; why.append("no_crash")
    else:
        pts -= 1; why.append("crash")
    if (r.get("dist200_pct") or 0) >= 5:
        pts += 1; why.append("above200")
    if r.get("earnings_in_days") is not None and r["earnings_in_days"] <= 5:
        pts -= 1; why.append("earnings_soon")
    return _clamp(pts), why


def score_pullback(r: dict) -> tuple:
    pts, why = 1, []
    if (r.get("rs_pct") or 0) >= 95:
        pts += 1; why.append("rs95")
    if (r.get("rvol") or 9) < 1:
        pts += 1; why.append("quiet")
    if r.get("near_ma") in ("ema10", "ema20"):
        pts += 1; why.append("short_ma")
    if (r.get("adr_pct") or 9) <= 4:
        pts += 1; why.append("tight")
    if r.get("earnings_in_days") is not None and r["earnings_in_days"] <= 5:
        pts -= 1; why.append("earnings_soon")
    return _clamp(pts), why


SCORERS = {"ep": score_ep, "rsi2": score_rsi2, "pullback": score_pullback}
