"""Forward-looking labels for the three new indicators (tops, bottoms, trend
continuation).

An indicator can only be judged against a clear definition of what it is trying
to find, so this module turns price history into three sets of yes/no (or
three-way) answers that describe what happened *after* each day. These answers
are the "truth" the indicators in forward_model.py are trained to predict.

Two rules keep the labels honest:

1. **No peeking.** Every label for day ``t`` is worked out only from prices on
   day ``t`` and later. A label is therefore unknown (``NaN`` / sentinel) for
   the final days of the history, exactly the days whose outcome has not
   happened yet. This is deliberate: at "today" the label is unknown, which is
   why we need an indicator to estimate it.
2. **Size is measured in ATRs.** A move counts as big only relative to the
   stock's own normal daily swing, the Average True Range (ATR). A 3% fall is a
   large move for a calm utility and an ordinary wiggle for a jumpy chip maker,
   so a fixed percentage would mislabel both. Using ATR puts every stock on the
   same footing.

The definitions here are the ones described to the user:

* **Top**  - day ``t`` is the highest close in a +/-``window`` day neighbourhood
  AND the price then falls at least ``drop_atr`` ATRs below that close before it
  ever climbs ``drop_atr`` ATRs above it, within ``horizon`` days.
* **Bottom** - the mirror image.
* **Continuation / reversal / stall** - a triple-barrier test. From day ``t``
  put one line ``barrier_atr`` ATRs above the close and one the same distance
  below, and watch for ``horizon`` days. If the line in the trend's direction is
  reached first the trend *continued*; the opposite line first means it
  *reversed*; if neither is touched in time the trend *stalled*.
"""
from __future__ import annotations

import numpy as np

from .indicators import atr_filled
from .turning_points import rolling_window

# Sentinel for "trend label not applicable / not yet known".
STALL = 0
CONTINUE = 1
REVERSE = -1
UNKNOWN = np.nan


def _atr(df, w: int = 14) -> np.ndarray:
    """ATR in price units, with the warm-up days filled so early history is
    usable (atr_filled averages the true range it has so far)."""
    return atr_filled(df["High"].to_numpy(float), df["Low"].to_numpy(float),
                       df["Close"].to_numpy(float), w)


def label_tops_bottoms(df, window: int = 10, drop_atr: float = 2.0, horizon: int = 40,
                       atr_w: int = 14):
    """Mark confirmed tops and bottoms.

    Returns ``(is_top, is_bottom, known_until)`` where ``is_top`` and
    ``is_bottom`` are float arrays of 1.0 / 0.0 with ``NaN`` on days whose
    outcome is not yet decidable, and ``known_until`` is the last index whose
    label is trustworthy (``n - 1 - max(window, horizon)``).

    A day is a *top* when its close is the highest in the ``window`` days on each
    side **and** within the next ``horizon`` days the price first drops
    ``drop_atr`` ATRs below that close rather than first rising the same amount
    above it. "Rising above" matters: a high that is immediately exceeded was not
    really a top, just a step on the way up.
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    n = len(close)
    atr = _atr(df, atr_w)
    is_top = np.full(n, np.nan)
    is_bottom = np.full(n, np.nan)

    # Local peaks / troughs on the close (two-sided, so known only window days later).
    piv = rolling_window(close, window)
    peaks = set(piv.loc[piv["kind"] == "P", "idx"].astype(int))
    troughs = set(piv.loc[piv["kind"] == "T", "idx"].astype(int))

    last = n - 1 - max(window, horizon)
    for t in range(0, max(last + 1, 0)):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        hi_end = min(t + horizon, n - 1)
        # forward window t+1 .. hi_end
        fwd_hi = high[t + 1:hi_end + 1]
        fwd_lo = low[t + 1:hi_end + 1]
        if fwd_hi.size == 0:
            continue
        # ----- top -----
        is_top[t] = 0.0
        if t in peaks:
            up_amt = drop_atr * a
            # first time price rises up_amt above close vs first time it falls up_amt below
            down_hit = np.where(fwd_lo <= close[t] - up_amt)[0]
            up_hit = np.where(fwd_hi >= close[t] + up_amt)[0]
            d = down_hit[0] if down_hit.size else np.inf
            u = up_hit[0] if up_hit.size else np.inf
            if d < u:
                is_top[t] = 1.0
        # ----- bottom -----
        is_bottom[t] = 0.0
        if t in troughs:
            amt = drop_atr * a
            up_hit = np.where(fwd_hi >= close[t] + amt)[0]
            down_hit = np.where(fwd_lo <= close[t] - amt)[0]
            u = up_hit[0] if up_hit.size else np.inf
            d = down_hit[0] if down_hit.size else np.inf
            if u < d:
                is_bottom[t] = 1.0
    return is_top, is_bottom, last


def label_continuation(df, state: np.ndarray, barrier_atr: float = 2.0, horizon: int = 20,
                       atr_w: int = 14):
    """Triple-barrier label for "does the current trend continue?".

    ``state`` is the trend label per day (+1 up, -1 down, 0 sideways), as
    produced by :func:`atrade.trend.trend_state`.

    Returns ``(label, known_until)``. ``label`` is a float array taking
    ``CONTINUE`` (1), ``REVERSE`` (-1) or ``STALL`` (0) on days where the trend
    is up or down, and ``NaN`` where the trend is sideways (no direction to
    continue) or where the outcome is not yet decidable.

    From the close on day ``t`` we draw a line ``barrier_atr`` ATRs above and one
    the same distance below. Over the next ``horizon`` days the first line the
    price reaches decides the label: the trend-direction line means the trend
    continued, the opposite line means it reversed, and touching neither in time
    means it stalled. Barriers are checked on the daily high and low so an
    intraday reach counts.
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    n = len(close)
    atr = _atr(df, atr_w)
    state = np.asarray(state, float)
    out = np.full(n, np.nan)
    last = n - 1 - horizon
    for t in range(0, max(last + 1, 0)):
        st = state[t]
        a = atr[t]
        if st == 0 or not np.isfinite(st) or not np.isfinite(a) or a <= 0:
            continue
        hi_end = min(t + horizon, n - 1)
        fwd_hi = high[t + 1:hi_end + 1]
        fwd_lo = low[t + 1:hi_end + 1]
        if fwd_hi.size == 0:
            continue
        up = close[t] + barrier_atr * a
        dn = close[t] - barrier_atr * a
        up_hit = np.where(fwd_hi >= up)[0]
        dn_hit = np.where(fwd_lo <= dn)[0]
        u = up_hit[0] if up_hit.size else np.inf
        d = dn_hit[0] if dn_hit.size else np.inf
        if not np.isfinite(u) and not np.isfinite(d):
            out[t] = STALL
        elif u == d:
            out[t] = STALL          # both lines reached the same day: treat as indecisive
        else:
            winner_up = u < d
            trend_up = st > 0
            out[t] = CONTINUE if winner_up == trend_up else REVERSE
    return out, last


def base_rates(df, state, **kw) -> dict:
    """Convenience: the share of days that are tops / bottoms, and the
    continue / reverse / stall mix, over the labelable part of the history.
    Used to show how much better than chance an indicator does."""
    is_top, is_bot, _ = label_tops_bottoms(df, **{k: v for k, v in kw.items()
                                                   if k in ("window", "drop_atr", "horizon", "atr_w")})
    cont, _ = label_continuation(df, state, **{k: v for k, v in kw.items()
                                               if k in ("barrier_atr", "horizon", "atr_w")})
    out = {
        "top_rate": float(np.nanmean(is_top)) if np.isfinite(is_top).any() else np.nan,
        "bottom_rate": float(np.nanmean(is_bot)) if np.isfinite(is_bot).any() else np.nan,
    }
    c = cont[np.isfinite(cont)]
    if c.size:
        out["continue_rate"] = float(np.mean(c == CONTINUE))
        out["reverse_rate"] = float(np.mean(c == REVERSE))
        out["stall_rate"] = float(np.mean(c == STALL))
    return out
