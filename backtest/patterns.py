"""Indicators, the four pattern signals and trade simulation on daily bars.

Rules follow the agreed skill "halal-swing-scanner-rules". Daily-bar
approximations (no intraday data):
  * intraday breakout entries -> price crosses the trigger level next day,
    filled at max(open, trigger) plus slippage;
  * "stop at low of day" -> low of the bar before entry;
  * after the entry day, a bar that touches the stop is assumed stopped out.
"""
import numpy as np
import pandas as pd

SLIPPAGE = 0.0005  # 0.05 % per side
MAX_HOLD = 60
RSI2_MAX_HOLD = 10


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    for n in (5, 10, 50, 150, 200):
        d[f"sma{n}"] = c.rolling(n).mean()
    d["ema10"] = c.ewm(span=10, adjust=False).mean()
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["vol20"] = v.rolling(20).mean()
    d["vol50"] = v.rolling(50).mean()
    d["dvol20"] = (c * v).rolling(20).mean()
    d["adr"] = ((h / l - 1) * 100).rolling(20).mean()
    d["hi252"] = h.rolling(252).max()
    d["lo252"] = l.rolling(252).min()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    d["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d.loc[dn == 0, "rsi2"] = 100.0
    return d


def base_filter(d: pd.DataFrame, adr_max: float) -> pd.Series:
    return (
        (d["Close"] >= 10)
        & (d["vol20"] >= 1_000_000)
        & (d["dvol20"] >= 20_000_000)
        & (d["adr"] >= 2)
        & (d["adr"] <= adr_max)
    )


# ---------------------------------------------------------------- signals
def sig_pullback(d: pd.DataFrame) -> pd.Series:
    c = d["Close"]
    trend = (
        (c > d["sma50"])
        & (d["sma50"] > d["sma150"])
        & (d["sma150"] > d["sma200"])
        & (d["sma200"] > d["sma200"].shift(20))
        & (c >= 1.30 * d["lo252"])
        & (c >= 0.75 * d["hi252"])
    )
    near = pd.Series(False, index=d.index)
    for ma in ("ema10", "ema20", "sma50"):
        near |= (d["Low"] <= d[ma] * 1.02) & (c >= d[ma] * 0.98)
    pulled_back = c <= 0.97 * d["High"].rolling(10).max()
    quiet = d["Volume"].rolling(3).mean() < d["vol50"]
    return trend & near & pulled_back & quiet & base_filter(d, 6)


def sig_breakout(d: pd.DataFrame) -> pd.Series:
    c = d["Close"]
    prior_move = d["High"].rolling(60).max() >= 1.30 * d["Low"].rolling(60).min()
    rng_hi = d["High"].rolling(10).max()
    rng_lo = d["Low"].rolling(10).min()
    tight = (rng_hi / rng_lo - 1) <= 0.12
    higher_lows = d["Low"].rolling(5).min() >= d["Low"].shift(5).rolling(5).min()
    near_ma = (c / d["sma10"] - 1).abs() <= 0.02
    ready = c >= 0.97 * rng_hi
    return prior_move & tight & higher_lows & near_ma & ready & base_filter(d, 8)


def sig_ep(d: pd.DataFrame) -> pd.Series:
    c, o, h, l = d["Close"], d["Open"], d["High"], d["Low"]
    prev_c = c.shift(1)
    gap = (o > d["High"].shift(1)) & (c >= 1.10 * prev_c)
    vol = d["Volume"] >= 3 * d["vol50"].shift(1)
    strong_close = c >= l + 0.5 * (h - l)
    # liquidity / ADR measured on the day before the gap (gap day inflates them)
    pre = base_filter(d.shift(1), 8).fillna(False).astype(bool)
    return gap & vol & strong_close & pre


def sig_rsi2(d: pd.DataFrame) -> pd.Series:
    return (d["Close"] > d["sma200"]) & (d["rsi2"] < 10) & base_filter(d, 6)


# ------------------------------------------------------------- simulation
def _manage(a, i_entry, entry, stop, trigger):
    """Qullamaggie-style management: 1/3 at close of day 3 if in profit and
    stop to breakeven; rest trails on first close below SMA10."""
    o, h, l, c, sma10 = a["o"], a["h"], a["l"], a["c"], a["sma10"]
    risk = entry - stop
    remaining, realized, partial = 1.0, 0.0, False
    n = len(c)
    j = i_entry
    while j < n:
        if j == i_entry:
            # order of the entry-day low is unknown: if the bar opened above the
            # trigger, the low came after entry; otherwise count the stop only
            # when the day closed below it
            hit = l[j] <= stop if o[j] >= trigger else c[j] < stop
        else:
            hit = l[j] <= stop
        if hit:
            px = stop if j == i_entry else min(o[j], stop)
            px *= 1 - SLIPPAGE
            realized += remaining * (px - entry) / risk
            return realized, j
        held = j - i_entry
        if not partial and held >= 3 and c[j] > entry:
            px = c[j] * (1 - SLIPPAGE)
            realized += (1 / 3) * (px - entry) / risk
            remaining -= 1 / 3
            stop = max(stop, entry)
            partial = True
        elif partial and c[j] < sma10[j]:
            px = c[j] * (1 - SLIPPAGE)
            realized += remaining * (px - entry) / risk
            return realized, j
        if held >= MAX_HOLD:
            px = c[j] * (1 - SLIPPAGE)
            realized += remaining * (px - entry) / risk
            return realized, j
        j += 1
    return None, None  # still open at end of data


def _trade(a, i_entry, entry, stop, adr_pct, trigger):
    risk = entry - stop
    if risk <= 0 or risk / entry < 0.005:
        return None
    if risk / entry * 100 > adr_pct:  # stop wider than 1 ADR -> skip trade
        return None
    r, i_exit = _manage(a, i_entry, entry, stop, trigger)
    if r is None:
        return None
    return {"entry_i": i_entry, "exit_i": i_exit, "entry": entry, "stop": stop, "R": r}


def simulate(d: pd.DataFrame, start: str):
    a = {k: d[col].to_numpy() for k, col in
         (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"), ("sma10", "sma10"),
          ("sma5", "sma5"))}
    adr = d["adr"].to_numpy()
    n = len(d)
    first = int(np.searchsorted(d.index.values, np.datetime64(start)))
    trades = []

    def add(pattern, sig_i, t):
        if t:
            t.update(pattern=pattern, signal_date=d.index[sig_i],
                     entry_date=d.index[t["entry_i"]], exit_date=d.index[t["exit_i"]],
                     days=t["exit_i"] - t["entry_i"])
            trades.append(t)
            return t["exit_i"]
        return None

    # 1) pullback: entry next day above signal-day high, stop under pullback low
    busy = -1
    for i in np.flatnonzero(sig_pullback(d).to_numpy()):
        if i < first or i <= busy or i + 1 >= n:
            continue
        if a["h"][i + 1] > a["h"][i]:
            entry = max(a["o"][i + 1], a["h"][i]) * (1 + SLIPPAGE)
            stop = a["l"][max(0, i - 4): i + 1].min()
            busy = add("pullback", i, _trade(a, i + 1, entry, stop, adr[i], a["h"][i])) or busy

    # 2) breakout: entry next day above 10-day range high, stop prior-day low
    busy = -1
    rng_hi = d["High"].rolling(10).max().to_numpy()
    for i in np.flatnonzero(sig_breakout(d).to_numpy()):
        if i < first or i <= busy or i + 1 >= n:
            continue
        if a["h"][i + 1] > rng_hi[i]:
            entry = max(a["o"][i + 1], rng_hi[i]) * (1 + SLIPPAGE)
            busy = add("breakout", i, _trade(a, i + 1, entry, a["l"][i], adr[i], rng_hi[i])) or busy

    # 3) EP: within 5 days, entry on break of gap-day high, stop prior-day low
    busy = -1
    for g in np.flatnonzero(sig_ep(d).to_numpy()):
        if g < first or g <= busy:
            continue
        for k in range(g + 1, min(g + 6, n)):
            if a["c"][k - 1] < a["l"][g]:
                break
            if a["h"][k] > a["h"][g]:
                entry = max(a["o"][k], a["h"][g]) * (1 + SLIPPAGE)
                busy = add("ep", g, _trade(a, k, entry, a["l"][k - 1], adr[g - 1], a["h"][g])) or busy
                break

    # 4) RSI-2: entry at close, stop under signal-bar low, exit close > SMA5
    busy = -1
    for i in np.flatnonzero(sig_rsi2(d).to_numpy()):
        if i < first or i <= busy or i + 1 >= n:
            continue
        entry = a["c"][i] * (1 + SLIPPAGE)
        stop = a["l"][i]
        risk = entry - stop
        if risk <= 0 or risk / entry < 0.005 or risk / entry * 100 > adr[i]:
            continue
        r, j_exit = None, None
        for j in range(i + 1, min(i + 1 + RSI2_MAX_HOLD, n)):
            if a["l"][j] <= stop:
                px = min(a["o"][j], stop)
            elif a["c"][j] > a["sma5"][j] or j == i + RSI2_MAX_HOLD:
                px = a["c"][j]
            else:
                continue
            r, j_exit = (px * (1 - SLIPPAGE) - entry) / risk, j
            break
        if r is not None:
            trades.append({"pattern": "rsi2", "signal_date": d.index[i],
                           "entry_date": d.index[i], "exit_date": d.index[j_exit],
                           "days": j_exit - i, "entry": entry, "stop": stop, "R": r,
                           "entry_i": i, "exit_i": j_exit})
            busy = j_exit
    return trades
