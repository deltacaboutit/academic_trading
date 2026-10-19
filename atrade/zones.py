"""Framework 4a - support and resistance zones (book ch. 4, eqs. 4.7-4.14)."""
from __future__ import annotations

import numpy as np

from .signals import Signal
from .turning_points import rolling_window


def build_bins(prices: np.ndarray, x: float):
    """Percentage bins with the lowest/highest turning points at bin CENTRES.

    Returns (bounds, counts, x_hat). bounds has n_hat + 1 entries.
    """
    prices = np.asarray(prices, float)
    L1 = prices.min() / (1 + x / 2)
    L2 = prices.max() * (1 + x / 2)
    n = np.log(L2 / L1) / np.log(1 + x)
    n_hat = max(1, int(round(n)))
    x_hat = (L2 / L1) ** (1 / n_hat) - 1
    bounds = L1 * (1 + x_hat) ** np.arange(n_hat + 1)
    counts, _ = np.histogram(prices, bins=bounds)
    return bounds, counts, x_hat


def zones_from_points(prices: np.ndarray, x: float, min_points: int):
    """List of (low, high, level, count) for bins holding >= min_points turning points."""
    if len(prices) < min_points:
        return []
    bounds, counts, _ = build_bins(prices, x)
    out = []
    for j, c in enumerate(counts):
        if c >= min_points:
            lo, hi = bounds[j], bounds[j + 1]
            out.append((float(lo), float(hi), float((lo + hi) / 2), int(c)))
    return out


def zone_analysis(close: np.ndarray, s: dict):
    """Recompute zones every day from turning points known up to day t-1 and
    record hits, bounces and failures.

    Returns (signals, events, zones_by_day) where zones_by_day[t] is the list of
    zones in force on day t (only stored for the last day and on change days).
    """
    w, x = int(s["zone_w"]), float(s["zone_x"])
    mp = int(s["zone_min_points"])
    start = int(s["zone_start_days"])
    live = int(s["live_days_zone"])
    n = len(close)
    tps = rolling_window(close, w)
    known = tps["known"].to_numpy() if not tps.empty else np.array([])
    tp_prices = tps["price"].to_numpy() if not tps.empty else np.array([])

    signals: list[Signal] = []
    events: list[dict] = []
    zones_today: list = []
    last_count = -1
    history: dict[int, list] = {}
    tracking = None   # dict(lo, hi, came_from, start)

    for t in range(max(start, 1), n):
        cnt = int((known <= t - 1).sum())
        if cnt != last_count:
            zones_today = zones_from_points(tp_prices[known <= t - 1], x, mp)
            last_count = cnt
            history[t] = zones_today
        c = close[t]
        if np.isnan(c):
            continue
        if tracking is None:
            for (lo, hi, lvl, k) in zones_today:
                if lo <= c <= hi:
                    # direction: most recent close outside this zone
                    j = t - 1
                    while j >= 0 and (np.isnan(close[j]) or lo <= close[j] <= hi):
                        j -= 1
                    if j < 0:
                        break
                    came = "above" if close[j] > hi else "below"
                    tracking = {"lo": lo, "hi": hi, "level": lvl, "came": came, "start": t, "count": k}
                    events.append({"idx": t, "event": "hit",
                                   "zone": "support" if came == "above" else "resistance", "level": lvl})
                    break
        else:
            lo, hi = tracking["lo"], tracking["hi"]
            if c > hi or c < lo:
                left = "above" if c > hi else "below"
                ztype = "support" if tracking["came"] == "above" else "resistance"
                if left == tracking["came"]:
                    d = +1 if ztype == "support" else -1
                    wt = float(s["wt_support"] if ztype == "support" else s["wt_resistance"])
                    signals.append(Signal(f"{ztype.capitalize()} bounce", "reversal", "zone", d, t, weight=wt,
                                          live_until=t + live - 1,
                                          line=(0.0, lo if d > 0 else hi),
                                          text=f"Price bounced off the {ztype} zone near {tracking['level']:.2f}.",
                                          extra={"zone_lo": lo, "zone_hi": hi, "hit_day": tracking["start"]}))
                    events.append({"idx": t, "event": "bounce", "zone": ztype, "level": tracking["level"]})
                else:
                    events.append({"idx": t, "event": "failure", "zone": ztype, "level": tracking["level"],
                                   "note": "broken support becomes resistance" if ztype == "support"
                                   else "broken resistance becomes support"})
                tracking = None
    history[n - 1] = zones_today
    return signals, events, history
