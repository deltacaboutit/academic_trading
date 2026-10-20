"""Framework 4c - overbought / oversold gauges: RSI, Bollinger Bands, ROC."""
from __future__ import annotations

import numpy as np

from . import indicators as ind
from .signals import Signal
from .turning_points import rolling_window


def roc_levels(r: np.ndarray, train: int, tp_w: int):
    """Adaptive ROC levels (book eqs. 7.22-7.24).

    For day t: take the ROC turning points (rolling window tp_w) that lie in the
    past `train` days AND are already confirmed by day t; ul = mean of peaks,
    dl = mean of troughs.
    """
    n = len(r)
    ul = np.full(n, np.nan)
    dl = np.full(n, np.nan)
    rr = np.where(np.isnan(r), np.nanmean(r) if np.isfinite(np.nanmean(r)) else 0, r)
    tps = rolling_window(rr, tp_w)
    first_valid = int(np.argmax(~np.isnan(r))) if (~np.isnan(r)).any() else n
    tps = tps[tps["idx"] >= first_valid + tp_w]  # ignore points touching the NaN start
    if tps.empty:
        return ul, dl
    idx = tps["idx"].to_numpy(); price = tps["price"].to_numpy(); kind = tps["kind"].to_numpy()
    known = tps["known"].to_numpy()
    # A turning point counts on day t when it is confirmed (known <= t) and still inside the
    # training window (idx >= t - train + 1). It enters on day `known` and leaves on day
    # idx + train. Running sums of these entries/exits give the averages for every day at once.
    start_t = first_valid + train - 1
    for which, out in (("P", ul), ("T", dl)):
        m = kind == which
        enter = known[m]; leave = idx[m] + train; val = price[m]
        ok = leave > enter
        enter, leave, val = enter[ok], leave[ok], val[ok]
        cnt = np.zeros(n + 1); tot = np.zeros(n + 1)
        np.add.at(cnt, np.minimum(enter, n), 1); np.add.at(tot, np.minimum(enter, n), val)
        np.add.at(cnt, np.minimum(leave, n), -1); np.add.at(tot, np.minimum(leave, n), -val)
        c, s_ = np.cumsum(cnt)[:n], np.cumsum(tot)[:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(c > 0, s_ / np.where(c > 0, c, 1), np.nan)
        out[start_t:] = avg[start_t:]
    return ul, dl


def gauge_values(close: np.ndarray, s: dict) -> dict:
    r = ind.roc(close, int(s["roc_w"]))
    ul, dl = roc_levels(r, int(s["roc_train"]), int(s["roc_tp_w"]))
    bb = ind.bollinger(close, int(s["bb_w"]), float(s["bb_k"]))
    return {"rsi": ind.rsi(close, int(s["rsi_w"])), "bb": bb, "roc": r, "roc_ul": ul, "roc_dl": dl}


def gauge_signals(close: np.ndarray, s: dict, vals: dict | None = None) -> list[Signal]:
    vals = vals or gauge_values(close, s)
    live = int(s["live_days_indicator"])
    wt = float(s["wt_gauge"])
    out: list[Signal] = []

    def add(name, days, d, text):
        for t in days:
            out.append(Signal(name, "reversal", "gauge", d, int(t), weight=wt,
                              live_until=int(t) + live - 1, text=text))

    # RSI: buy when RSI crosses 30 from below, sell when it crosses 70 from above
    r = vals["rsi"]; lo, hi = float(s["rsi_low"]), float(s["rsi_high"])
    r1 = ind.shift(r)
    with np.errstate(invalid="ignore"):
        add(f"RSI {s['rsi_w']}", np.where((r > lo) & (r1 < lo))[0], +1, f"RSI rose back above {lo:.0f} (oversold).")
        add(f"RSI {s['rsi_w']}", np.where((r < hi) & (r1 > hi))[0], -1, f"RSI fell back below {hi:.0f} (overbought).")

    # Bollinger Bands (book eq. 7.17-7.18); only the first day of a run counts
    bb = vals["bb"]; up, lw = bb["up"], bb["low"]
    p1, up1, lw1 = ind.shift(close), ind.shift(up), ind.shift(lw)
    with np.errstate(invalid="ignore"):
        buy = ((close > up) & (p1 > up1)) | ((close > lw) & (p1 < lw1))
        sell = ((close < lw) & (p1 < lw1)) | ((close < up) & (p1 > up1))
    for arr, d, txt in ((buy, +1, "Price crossed back above the lower Bollinger band, or kept closing above the upper band."),
                        (sell, -1, "Price crossed back below the upper Bollinger band, or kept closing below the lower band.")):
        first = arr & ~np.concatenate([[False], arr[:-1]])
        add(f"Bollinger {s['bb_w']}/{s['bb_k']}", np.where(first)[0], d, txt)

    # ROC with adaptive levels
    rc, ul, dl = vals["roc"], vals["roc_ul"], vals["roc_dl"]
    with np.errstate(invalid="ignore"):
        b = (rc > dl) & (ind.shift(rc) < ind.shift(dl))
        sl = (rc < ul) & (ind.shift(rc) > ind.shift(ul))
    add(f"ROC {s['roc_w']}", np.where(b)[0], +1, "Rate of change rose back above its oversold level.")
    add(f"ROC {s['roc_w']}", np.where(sl)[0], -1, "Rate of change fell back below its overbought level.")
    return out
