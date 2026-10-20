"""Framework 4d - similarity search with Dynamic Time Warping (book ch. 9).

An unscored context panel: it never votes in the rating (the book presents it as a
research idea, not a proven signal).

1. Smooth the price history with a Savitzky-Golay filter: a cubic fitted to each 21-day
   window; the first and last 10 days use the cubic fitted to the first / last 21 days.
2. Slopes ("derivative DTW", Keogh and Pazzani 2001):
       p'_m = ((p_m - p_{m-1}) + (p_{m+1} - p_{m-1}) / 2) / 2,  ends copy their neighbour.
3. Standardise the slope series (mean 0, sd 1). The query Q is the last `query_days`
   (about 2 months = 61 days); the history Y is everything before it.
4. Subsequence DTW: d(n, m) = (q_n - y_m)^2,
       D(n, m) = d(n, m) + min{D(n-1, m-1), D(n-1, m), D(n, m-1)},  D(1, m) = d(1, m),
   so a match may start on any day. Optional Sakoe-Chiba band: the matched stretch may not
   run more than `band` days ahead of or behind the query.
5. D(N, m) is the cost of the best match ending on day m. Days with cost below tau
   (15 in the book) are match ends; after each, the next 20 days are skipped. The start
   day comes from tracing the cheapest path back. The threshold applies to the raw
   accumulated cost (not divided by the path length).
   tau = 15 comes from the book's S&P 500 example with a constructed query; real two-month
   queries often have no match that close, so the panel also lists the `top` closest
   non-overlapping matches whatever their cost (tool choice), clearly marked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def slopes(close: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    p = savgol_filter(np.asarray(close, float), window, poly, mode="interp")
    d = np.empty_like(p)
    d[1:-1] = ((p[1:-1] - p[:-2]) + (p[2:] - p[:-2]) / 2) / 2
    d[0], d[-1] = d[1], d[-2]
    return d


def subsequence_dtw(Q: np.ndarray, Y: np.ndarray, band: int | None = None):
    """Last-row costs D(N, m) and the start day of the best path ending at each m."""
    N, M = len(Q), len(Y)
    INF = np.inf
    prev = (Q[0] - Y) ** 2                       # row 1: a match may start anywhere
    prev_start = np.arange(M)
    for n in range(1, N):
        dn = (Q[n] - Y) ** 2
        cur = np.full(M, INF); cur_start = np.zeros(M, dtype=int)
        for m in range(M):
            best, s = prev[m], prev_start[m]                       # D(n-1, m)
            if m > 0:
                if prev[m - 1] < best:
                    best, s = prev[m - 1], prev_start[m - 1]       # D(n-1, m-1)
                if cur[m - 1] < best:
                    best, s = cur[m - 1], cur_start[m - 1]         # D(n, m-1)
            if band is not None and abs((m - s) - n) > band:
                continue
            cur[m] = dn[m] + best; cur_start[m] = s
        prev, prev_start = cur, cur_start
    return prev, prev_start


def _row(close, start, m, c, horizons, kind):
    n = len(close)
    r = {"start_idx": int(start), "end_idx": int(m), "cost": float(c), "kind": kind}
    for h in horizons:
        r[f"ret_{h}d"] = float(np.log(close[m + h] / close[m])) if m + h < n else np.nan
    return r


def find_matches(close: np.ndarray, query_days: int = 61, tau: float = 15.0, skip: int = 20,
                 band: int | None = None, horizons=(5, 10, 20, 40), top: int = 10) -> dict:
    close = np.asarray(close, float)
    n = len(close)
    if n < query_days * 3:
        return {"matches": pd.DataFrame(), "summary": pd.DataFrame(), "query": (n - query_days, n - 1)}
    z = slopes(close)
    z = (z - z.mean()) / z.std()
    Q, Y = z[n - query_days:], z[:n - query_days]
    cost, start = subsequence_dtw(Q, Y, band)
    cost = cost.copy()
    cost[max(0, len(Y) - skip):] = np.inf        # a match must end at least `skip` days before the query
    rows = []
    m = 0
    while m < len(Y):
        if cost[m] < tau:
            rows.append(_row(close, start[m], m, cost[m], horizons, f"below threshold {tau:g}"))
            m += skip + 1
        else:
            m += 1
    if len(rows) < top:                           # closest matches that do not overlap each other
        taken = np.zeros(len(Y), bool)
        for r in rows:
            taken[r["start_idx"]:r["end_idx"] + skip + 1] = True
        for m in np.argsort(cost):
            if len(rows) >= top or not np.isfinite(cost[m]):
                break
            if taken[start[m]:m + 1].any():
                continue
            rows.append(_row(close, start[m], m, cost[m], horizons, "closest"))
            taken[max(0, start[m] - skip):m + skip + 1] = True
    matches = pd.DataFrame(rows).sort_values("cost").reset_index(drop=True) if rows else pd.DataFrame()
    summ = []
    lr_all = np.log(close)
    for h in horizons:
        base = lr_all[h:] - lr_all[:-h]
        vals = matches[f"ret_{h}d"].dropna() if len(matches) else pd.Series(dtype=float)
        summ.append({"horizon_days": h, "matches": int(len(vals)),
                     "avg_return_after_matches": float(vals.mean()) if len(vals) else np.nan,
                     "share_up": float((vals > 0).mean()) if len(vals) else np.nan,
                     "avg_return_all_days": float(base.mean()), "share_up_all_days": float((base > 0).mean())})
    return {"matches": matches, "summary": pd.DataFrame(summ), "query": (n - query_days, n - 1),
            "best_cost": float(np.min(cost)) if len(cost) else np.nan}
