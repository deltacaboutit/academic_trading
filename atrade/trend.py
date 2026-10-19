"""Framework 3 - trend confirmation (book ch. 1, 5 and 7)."""
from __future__ import annotations

import numpy as np

from . import indicators as ind
from .signals import Signal

MAX_PRICE_FILTER_WAIT = 10   # days to wait for a price filter to be met (tool choice)


def _filter(t, d, A, B, s, close, low, high, filter_ok):
    """Apply the chosen false-signal filter to a raw cross on day t in direction d.

    Returns the day the signal is accepted, or None.
    filter_ok: which filters make sense for this rule ({'price', 'wholeday'}).
    If the chosen filter does not apply to a rule, the time filter is used, so
    every signal still has to pass one of the filters.
    """
    n = len(close)
    ftype = s["filter_type"]
    if ftype in ("price", "wholeday") and ftype not in filter_ok:
        ftype = "time"
    if ftype == "none":
        return t
    if ftype == "time":
        k = int(s["filter_time_k"])
        if t + k >= n:
            return None
        for m in range(1, k + 1):
            a, b = A[t + m], B[t + m]
            if np.isnan(a) or np.isnan(b) or (d > 0 and not a > b) or (d < 0 and not a < b):
                return None
        return t + k
    if ftype == "price":
        x = float(s["filter_price_x"])
        for u in range(t, min(n, t + MAX_PRICE_FILTER_WAIT + 1)):
            a, b = A[u], B[u]
            if np.isnan(a) or np.isnan(b):
                return None
            if (d > 0 and a <= b) or (d < 0 and a >= b):
                return None           # fell back across the line first
            if (d > 0 and a > b * (1 + x)) or (d < 0 and a < b * (1 - x)):
                return u
        return None
    if ftype == "wholeday":
        if t + 1 >= n:
            return None
        b = B[t + 1]
        if d > 0 and low[t + 1] > b:
            return t + 1
        if d < 0 and high[t + 1] < b:
            return t + 1
        return None
    return t


def trend_state(close: np.ndarray, s: dict) -> np.ndarray:
    """+1 uptrend, -1 downtrend, 0 sideways (ch. 5).

    Uptrend: slope of a straight line through the last 100 closes is positive AND
    more than 70% of those days closed above their 50-day average. Downtrend is
    the mirror image (slope negative and fewer than 30%). Anything else: sideways.
    """
    n = len(close)
    N = int(s["trend_state_days"])
    ma = ind.sma(close, int(s["trend_state_ma"]))
    above = (close > ma).astype(float)
    above[np.isnan(ma)] = np.nan
    out = np.zeros(n)
    x = np.arange(N, dtype=float)
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    for t in range(N - 1, n):
        seg = close[t - N + 1:t + 1]
        share_seg = above[t - N + 1:t + 1]
        if np.isnan(share_seg).any():
            continue
        slope = (xm * (seg - seg.mean())).sum() / denom
        share = share_seg.mean()
        if slope > 0 and share > s["trend_up_share"]:
            out[t] = 1
        elif slope < 0 and share < s["trend_down_share"]:
            out[t] = -1
    return out


def trend_signals(df, s: dict) -> list[Signal]:
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    high = df["High"].to_numpy(float)
    live = int(s["live_days_indicator"])
    out: list[Signal] = []

    def add_cross(name, A, Bbuy, Bsell, filter_ok, kind="crossover", text_buy="", text_sell=""):
        up = ind.cross_up(A, Bbuy)
        dn = ind.cross_down(A, Bsell)
        for t in np.where(up)[0]:
            e = _filter(t, +1, A, Bbuy, s, close, low, high, filter_ok)
            if e is not None:
                out.append(Signal(name, "trend", kind, +1, int(e), live_until=int(e) + live - 1,
                                  text=text_buy, extra={"cross_day": int(t)}))
        for t in np.where(dn)[0]:
            e = _filter(t, -1, A, Bsell, s, close, low, high, filter_ok)
            if e is not None:
                out.append(Signal(name, "trend", kind, -1, int(e), live_until=int(e) + live - 1,
                                  text=text_sell, extra={"cross_day": int(t)}))

    # Moving-average crossover 10/50
    a, b = int(s["mac_short"]), int(s["mac_long"])
    sa, sb = ind.sma(close, a), ind.sma(close, b)
    add_cross(f"MA crossover {a}/{b}", sa, sb, sb, {"price"},
              text_buy=f"The {a}-day average crossed above the {b}-day average.",
              text_sell=f"The {a}-day average crossed below the {b}-day average.")

    # MACD 12/26/9
    f, sl, sg = int(s["macd_fast"]), int(s["macd_slow"]), int(s["macd_signal"])
    m, sig = ind.macd(close, f, sl, sg)
    add_cross(f"MACD {f}/{sl}/{sg}", m, sig, sig, set(),
              text_buy="MACD crossed above its signal line.",
              text_sell="MACD crossed below its signal line.")

    # Momentum 12 with 9-day signal line
    w, w2 = int(s["mom_w"]), int(s["mom_signal"])
    mom, msl = ind.momentum(close, w, w2)
    add_cross(f"Momentum {w}/{w2}", mom, msl, msl, set(),
              text_buy=f"{w}-day momentum crossed above its signal line.",
              text_sell=f"{w}-day momentum crossed below its signal line.")

    # Range breakout: highest / lowest close of the past w days (not today)
    for w in s["breakout_ws"]:
        w = int(w)
        prev = ind.shift(close, 1)
        hh = ind.rolling_max(prev, w)
        ll = ind.rolling_min(prev, w)
        add_cross(f"Breakout {w}-day", close, hh, ll, {"price", "wholeday"}, kind="breakout",
                  text_buy=f"Close broke above the highest close of the past {w} days.",
                  text_sell=f"Close broke below the lowest close of the past {w} days.")

    # Price versus its own simple average
    for w in s["sma_ws"]:
        w = int(w)
        m = ind.sma(close, w)
        add_cross(f"Price vs SMA {w}", close, m, m, {"price", "wholeday"},
                  text_buy=f"Close crossed above its {w}-day average.",
                  text_sell=f"Close crossed below its {w}-day average.")
    return out


def alexander_signals(close: np.ndarray, x: float) -> list[Signal]:
    """Alexander's x% filter rule (benchmark only, no rating weight).

    Long after the price rises x% from its most recent low; short after it falls
    x% from its most recent high.
    """
    out: list[Signal] = []
    pos = 0
    lo = hi = close[0]
    for t in range(1, len(close)):
        p = close[t]
        if np.isnan(p):
            continue
        lo = min(lo, p); hi = max(hi, p)
        if pos <= 0 and p >= lo * (1 + x):
            pos = 1; hi = p
            out.append(Signal(f"Alexander {x:.0%} filter", "trend", "benchmark", +1, t,
                              weight=0.0, text="Rose x% from the latest low."))
        elif pos >= 0 and p <= hi * (1 - x):
            pos = -1; lo = p
            out.append(Signal(f"Alexander {x:.0%} filter", "trend", "benchmark", -1, t,
                              weight=0.0, text="Fell x% from the latest high."))
    return out
