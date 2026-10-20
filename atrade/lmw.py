"""[LMW] items - Lo, Mamaysky and Wang (2000), "Foundations of Technical Analysis".

These run ALONGSIDE the textbook items and never change them (framework source tag [LMW]).

Method C - kernel-regression turning points (framework 2):
  * every rolling window of 38 trading days (l = 35 plus lag d = 3) is smoothed with a
    Nadaraya-Watson regression, Gaussian kernel;
  * bandwidth: LMW use 0.3 x h*, where h* minimises the leave-one-out cross-validation
    error, but they do not publish the resulting width, and on daily prices the recipe is
    degenerate (the usual form picks almost no smoothing; equation 13 as printed picks the
    widest allowed). The tool therefore uses a FIXED width calibrated so that patterns are
    found as often as LMW report (about 8.3 per stock-year on NYSE/AMEX, 1962-96):
    1.8 days by default, re-calibrated on TRAINING data by the research runner
    (calibrate_bandwidth). The cross-validation version is kept as an option (h=None);
  * a peak (trough) is where the smoothed line's slope turns from + to - (- to +); the
    turning point is the highest (lowest) real price within one day either side;
  * a pattern counts only if its last turning point is on day 35 of the window (t+34),
    and it is detected 3 days later, at the window's end (t+37). Nothing later is used.

Ten patterns from five consecutive turning points E1..E5 (framework 4e): HS-LMW, IHS-LMW,
BTOP, BBOT, TTOP, TBOT, RTOP, RBOT, DTOP-LMW, DBOT-LMW.

They start in information-only mode (no vote). A direction can be given (from the
Framework 9b tests, or the usual direction for the four classical shapes); a directed
pattern votes when the close breaks its outer boundary within `break_days` days.

Volume trend (turnover): the average volume of the second half of the pattern's window
versus the first half (increasing if > 1.2x, decreasing if < 1/1.2). Turnover = volume /
shares outstanding; over a 38-day window the share count is effectively constant, so it
cancels out of this ratio (tool choice).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .signals import Signal

L, D = 35, 3
WIN = L + D                                      # 38
DEFAULT_H = 1.8                                  # days; see calibrate_bandwidth()
LMW_RATE = (1611 + 1654 + 725 + 748 + 1294 + 1193 + 1482 + 1616 + 2076 + 2075) / (350 * 5)  # per stock-year
H_GRID = np.exp(np.linspace(np.log(0.6), np.log(12.0), 24))
NAMES = ["HS-LMW", "IHS-LMW", "BTOP", "BBOT", "TTOP", "TBOT", "RTOP", "RBOT", "DTOP-LMW", "DBOT-LMW"]
CLASSIC_DIRECTION = {"HS-LMW": -1, "IHS-LMW": +1, "DTOP-LMW": -1, "DBOT-LMW": +1}


def _kernel(h):
    x = np.arange(WIN)
    return np.exp(-0.5 * ((x[:, None] - x[None, :]) / h) ** 2)


def smooth_windows(P: np.ndarray):
    """P: (N, 38) windows. Returns smoothed lines (N, 38) and chosen bandwidths."""
    N = len(P)
    cv = np.empty((len(H_GRID), N))
    for k, h in enumerate(H_GRID):
        W = _kernel(h); np.fill_diagonal(W, 0.0)          # leave-one-out
        m = (P @ W.T) / W.sum(axis=1)
        cv[k] = ((P - m) ** 2).mean(axis=1)
    hstar = H_GRID[np.argmin(cv, axis=0)]
    h = 0.3 * hstar
    out = np.empty_like(P)
    for hv in np.unique(h):
        sel = h == hv
        W = _kernel(hv)
        out[sel] = (P[sel] @ W.T) / W.sum(axis=1)
    return out, h


def window_extrema(price_win: np.ndarray, m: np.ndarray):
    """Turning points of one window: list of (index in window, price, 'P' or 'T')."""
    dm = np.diff(m)
    out = []
    for i in range(1, WIN - 1):
        a, b = dm[i - 1], dm[i]
        if a > 0 and b <= 0:
            j = i - 1 + int(np.argmax(price_win[i - 1:i + 2])); out.append((j, float(price_win[j]), "P"))
        elif a < 0 and b >= 0:
            j = i - 1 + int(np.argmin(price_win[i - 1:i + 2])); out.append((j, float(price_win[j]), "T"))
    # keep strictly alternating (merge equal neighbours: keep the more extreme)
    clean = []
    for e in out:
        if clean and clean[-1][2] == e[2]:
            if (e[2] == "P" and e[1] > clean[-1][1]) or (e[2] == "T" and e[1] < clean[-1][1]):
                clean[-1] = e
        else:
            clean.append(e)
    return clean


def _within(vals, pct):
    avg = np.mean(vals)
    return all(abs(v - avg) <= pct * avg for v in vals)


def classify(ext) -> list[str]:
    """Which LMW patterns the turning points of one window form. The eight five-point
    patterns use the LAST five turning points E1..E5; double tops/bottoms use all of them."""
    out = []
    if len(ext) >= 5:
        E1, E2, E3, E4, E5 = [e[1] for e in ext[-5:]]
        if ext[-5][2] == "P":                                     # E1 is a peak
            if E3 > E1 and E3 > E5 and _within([E1, E5], 0.015) and _within([E2, E4], 0.015):
                out.append("HS-LMW")
            if E1 < E3 < E5 and E2 > E4:
                out.append("BTOP")
            if E1 > E3 > E5 and E2 < E4:
                out.append("TTOP")
            if _within([E1, E3, E5], 0.0075) and _within([E2, E4], 0.0075) and min(E1, E3, E5) > max(E2, E4):
                out.append("RTOP")
        else:                                                     # E1 is a trough
            if E3 < E1 and E3 < E5 and _within([E1, E5], 0.015) and _within([E2, E4], 0.015):
                out.append("IHS-LMW")
            if E1 > E3 > E5 and E2 < E4:
                out.append("BBOT")
            if E1 < E3 < E5 and E2 > E4:
                out.append("TBOT")
            if _within([E1, E3, E5], 0.0075) and _within([E2, E4], 0.0075) and min(E2, E4) > max(E1, E3, E5):
                out.append("RBOT")
    if len(ext) >= 2:
        out += _double(ext)
    return out


def _double(ext) -> list[str]:
    """DTOP/DBOT use ALL turning points in the window: E1 is the first one; Ea is the
    highest later peak (lowest later trough); E1 and Ea within 1.5% of their average and
    more than 22 trading days apart."""
    e1 = ext[0]
    later = [e for e in ext[1:] if e[2] == e1[2]]
    if not later:
        return []
    ea = max(later, key=lambda e: e[1]) if e1[2] == "P" else min(later, key=lambda e: e[1])
    if _within([e1[1], ea[1]], 0.015) and ea[0] - e1[0] > 22:
        return ["DTOP-LMW" if e1[2] == "P" else "DBOT-LMW"]
    return []


def smooth_fixed(P: np.ndarray, h: float):
    W = _kernel(h)
    return (P @ W.T) / W.sum(axis=1), np.full(len(P), h)


def detect(close: np.ndarray, volume: np.ndarray | None = None, h: float | None = DEFAULT_H) -> pd.DataFrame:
    """All LMW pattern detections. One row per (pattern, detection day).
    h: fixed bandwidth in days, or None for the cross-validation recipe (0.3 x h*)."""
    close = np.asarray(close, float); n = len(close)
    if n < WIN + 1:
        return pd.DataFrame(columns=["pattern", "detect_idx", "start_idx", "E5_idx", "hi", "lo", "volume_trend", "h"])
    from numpy.lib.stride_tricks import sliding_window_view as swv
    P = swv(close, WIN).astype(float)
    ok = np.isfinite(P).all(axis=1)
    m = np.full_like(P, np.nan); hs = np.full(len(P), np.nan)
    m[ok], hs[ok] = smooth_windows(P[ok]) if h is None else smooth_fixed(P[ok], h)
    rows = []
    for t in np.where(ok)[0]:
        ext = window_extrema(P[t], m[t])
        if not ext or ext[-1][0] != L - 1:          # last turning point must be on day t+34
            continue
        for name in classify(ext):
            five = ext[-5:]
            vt = "n/a"
            if volume is not None:
                v = np.asarray(volume[t:t + WIN], float)
                v1, v2 = np.nanmean(v[:WIN // 2]), np.nanmean(v[WIN // 2:])
                vt = "increasing" if v2 > 1.2 * v1 else "decreasing" if v1 > 1.2 * v2 else "flat"
            rows.append({"pattern": name, "detect_idx": int(t + WIN - 1), "start_idx": int(t + five[0][0]),
                         "E5_idx": int(t + five[-1][0]), "hi": max(e[1] for e in five), "lo": min(e[1] for e in five),
                         "points": [(int(t + e[0]), e[1]) for e in five], "volume_trend": vt, "h": float(hs[t])})
    return pd.DataFrame(rows)


def to_signals(det: pd.DataFrame, close: np.ndarray, directions: dict | None = None,
               break_days: int = 20, live_days: int = 10) -> list[Signal]:
    """Directed LMW patterns -> signals on the first close beyond the outer boundary
    (above the highest turning point for a buy, below the lowest for a sell)."""
    directions = CLASSIC_DIRECTION if directions is None else directions
    out = []
    n = len(close)
    for r in det.itertuples(index=False):
        d = int(directions.get(r.pattern, 0))
        if d == 0:
            continue
        for e in range(r.detect_idx, min(n, r.detect_idx + break_days + 1)):
            if (d > 0 and close[e] > r.hi) or (d < 0 and close[e] < r.lo):
                out.append(Signal(f"[LMW] {r.pattern}", "reversal", "lmw", d, int(e), weight=1.0,
                                  live_until=int(e) + live_days - 1, vol_adjust=False,
                                  line=(0.0, float(r.hi if d > 0 else r.lo)),
                                  points=[(i, p, "") for i, p in r.points],
                                  text=f"LMW {r.pattern}: close broke the pattern's outer boundary.",
                                  extra={"volume_trend": r.volume_trend, "detect_idx": int(r.detect_idx)}))
                break
    return out


# ------------------------------------------------------- 9b information tests ---

def post_pattern_returns(det: pd.DataFrame, close: np.ndarray, a: int = 0, b: int | None = None):
    """Normalised one-day log returns that start 3 days after the pattern completes
    (the close of the detection day to the next close), plus all normalised returns
    in the same period."""
    b = len(close) if b is None else b
    lr = np.full(len(close), np.nan); lr[1:] = np.diff(np.log(close))
    seg = lr[a:b]; mu, sd = np.nanmean(seg), np.nanstd(seg)
    X = (lr - mu) / sd
    out = {}
    for name, g in det.groupby("pattern"):
        idx = g["detect_idx"].to_numpy() + 1
        idx = idx[(idx >= a) & (idx < b)]
        out[name] = X[idx][np.isfinite(X[idx])]
    uncond = X[a:b]; uncond = uncond[np.isfinite(uncond)]
    return out, uncond


def info_tests(cond: np.ndarray, uncond: np.ndarray) -> dict:
    """Decile chi-square test and Kolmogorov-Smirnov test (framework 9b, steps 3-4)."""
    n = len(cond)
    if n < 10 or len(uncond) < 100:
        return {"n": n, "Q": np.nan, "p_decile": np.nan, "KS": np.nan, "p_ks": np.nan, "mean": float(np.mean(cond)) if n else np.nan}
    cuts = np.quantile(uncond, np.linspace(0, 1, 11)[1:-1])
    counts = np.bincount(np.searchsorted(cuts, cond, side="right"), minlength=10)
    Q = float(((counts - 0.1 * n) ** 2 / (0.1 * n)).sum())
    n1, n2 = n, len(uncond)
    grid = np.sort(np.concatenate([cond, uncond]))
    F1 = np.searchsorted(np.sort(cond), grid, side="right") / n1
    F2 = np.searchsorted(np.sort(uncond), grid, side="right") / n2
    gamma = math.sqrt(n1 * n2 / (n1 + n2)) * float(np.max(np.abs(F1 - F2)))
    return {"n": n, "Q": Q, "p_decile": float(stats.chi2.sf(Q, 9)), "KS": gamma,
            "p_ks": float(stats.kstwobign.sf(gamma)), "mean": float(np.mean(cond))}


def info_table(det: pd.DataFrame, close: np.ndarray, a: int = 0, b: int | None = None) -> pd.DataFrame:
    cond, uncond = post_pattern_returns(det, close, a, b)
    rows = []
    for name in NAMES:
        r = info_tests(cond.get(name, np.array([])), uncond)
        r["pattern"] = name
        r["informative"] = bool((r["p_decile"] < 0.05) or (r["p_ks"] < 0.05)) if np.isfinite(r["p_decile"]) else None
        rows.append(r)
    return pd.DataFrame(rows)[["pattern", "n", "mean", "Q", "p_decile", "KS", "p_ks", "informative"]]


def calibrate_bandwidth(closes: list, target: float = LMW_RATE, grid=(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4)) -> dict:
    """Choose the fixed bandwidth whose pattern rate (all ten patterns, per stock-year)
    on the given TRAINING price series is closest to LMW's published rate."""
    rates = {}
    years = sum(len(c) for c in closes) / 252
    for h in grid:
        n = sum(len(detect(c, None, h)) for c in closes)
        rates[h] = n / years if years else np.nan
    best = min(rates, key=lambda h: abs(rates[h] - target))
    return {"h": best, "rates": rates, "target": target}


def learn_directions(table: pd.DataFrame) -> dict:
    """From the information tests on TRAINING data: an informative pattern votes in the
    direction its post-pattern returns lean; others stay information-only."""
    out = {}
    for r in table.itertuples(index=False):
        out[r.pattern] = int(np.sign(r.mean)) if (r.informative and np.isfinite(r.mean)) else 0
    return out
