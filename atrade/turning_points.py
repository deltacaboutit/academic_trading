"""Framework 2 - turning-point finder.

Method A: rolling window (book ch. 2, RW function). Strict inequalities on both
sides, so flat tops of equal prices are never marked.
Method B: Perceptually Important Points (PIPs) with Euclidean, perpendicular
or vertical distance.

A turning point at day t found with window w is only CONFIRMED on day t + w.
Every user of these points must respect that (field "known").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_window(p: np.ndarray, w: int) -> pd.DataFrame:
    """Return all local peaks and troughs.

    Columns: idx (day number), price, kind ('P' or 'T'), known (idx + w).
    Day t is a peak if p_t > max(p_{t-w..t-1}) and p_t > max(p_{t+1..t+w}).
    """
    p = np.asarray(p, float)
    n = len(p)
    rows = []
    if n < 2 * w + 1:
        return pd.DataFrame(columns=["idx", "price", "kind", "known"])
    # sliding views for speed
    from numpy.lib.stride_tricks import sliding_window_view as swv
    win = swv(p, w)                       # win[i] = p[i:i+w]
    left_max = np.full(n, np.nan); left_min = np.full(n, np.nan)
    right_max = np.full(n, np.nan); right_min = np.full(n, np.nan)
    wmax = np.nanmax(win, axis=1) if len(win) else np.array([])
    wmin = np.nanmin(win, axis=1) if len(win) else np.array([])
    # left window of t is p[t-w:t] = win[t-w]; right window is p[t+1:t+w+1] = win[t+1]
    t = np.arange(w, n - w)
    left_max[t] = wmax[t - w]; left_min[t] = wmin[t - w]
    right_max[t] = wmax[t + 1]; right_min[t] = wmin[t + 1]
    with np.errstate(invalid="ignore"):
        is_peak = (p > left_max) & (p > right_max)
        is_trough = (p < left_min) & (p < right_min)
    for i in np.where(is_peak)[0]:
        rows.append((int(i), float(p[i]), "P", int(i + w)))
    for i in np.where(is_trough)[0]:
        rows.append((int(i), float(p[i]), "T", int(i + w)))
    df = pd.DataFrame(rows, columns=["idx", "price", "kind", "known"])
    return df.sort_values("idx").reset_index(drop=True)


# ---------------------------------------------------------------- PIPs ----

def _dist(kind: str, xs, ys, x1, y1, x2, y2):
    if kind == "euclidean":
        return np.sqrt((x1 - xs) ** 2 + (y1 - ys) ** 2) + np.sqrt((x2 - xs) ** 2 + (y2 - ys) ** 2)
    s = (y2 - y1) / (x2 - x1)
    c = y1 - x1 * s
    v = np.abs(s * xs + c - ys)
    if kind == "vertical":
        return v
    return v / np.sqrt(s * s + 1)            # perpendicular (default)


def pips(y: np.ndarray, n_points: int, dist: str = "perpendicular") -> list[int]:
    """Return the positions (0-based, sorted) of n_points PIPs in y.

    Starts with the first and last point, then repeatedly adds the point with the
    largest distance from the line joining its two neighbouring PIPs.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return []
    chosen = [0, n - 1] if n > 1 else [0]
    while len(chosen) < min(n_points, n):
        chosen.sort()
        best_d, best_i = -1.0, None
        for a, b in zip(chosen[:-1], chosen[1:]):
            if b - a < 2:
                continue
            xs = np.arange(a + 1, b)
            d = _dist(dist, xs, y[a + 1:b], a, y[a], b, y[b])
            j = int(np.argmax(d))
            if d[j] > best_d:
                best_d, best_i = float(d[j]), int(xs[j])
        if best_i is None:
            break
        chosen.append(best_i)
    return sorted(chosen)
