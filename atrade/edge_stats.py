"""Statistics for the random-twin test (ported from PatternFinder 2.5's analyze.py).

Words used
  edge          a trade's return minus the average return of its random twins (see
                twins.py): the part the PATTERN adds. In % per trade.
  clustered standard error
                trades that start in the same calendar month share the same market
                moves, so they are not truly independent. The standard error treats
                each month as one group (a cluster-robust "sandwich" formula with the
                usual small-sample correction). This is the honest one to read.
  t-stat, p     how sure we can be that an edge is not luck. p is the chance of an
                edge this far from zero by luck alone, from Student's t distribution
                with (number of months - 1) degrees of freedom.
  holm_p, bh_q  when many patterns (or checks) are tested at once, a few look good by
                luck. These adjust each p-value for that (Holm 1979; Benjamini &
                Hochberg 1995). Below 0.05 is the usual bar.
  too few       a group with fewer than 20 trades: shown, but too small to judge.

A textbook check counts as HELPS only if all three hold (PatternFinder 2.5's standard):
  * trades that passed it beat trades that failed it, with Holm p < 0.05 across all checks,
  * the difference is above 0 in BOTH halves of the period,
  * at least 20 trades passed and 20 failed.
Only checks marked HELPS may go into confirm.PROVEN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

MIN_TRADES = 20   # below this a group is shown but labelled "too few" (PatternFinder 2.5)


def clustered_se(x, clusters) -> float:
    """Standard error of the mean of x, allowing x to be correlated inside each cluster."""
    x = pd.Series(np.asarray(x, float))
    n = len(x)
    if n < 2:
        return np.nan
    g = (x - x.mean()).groupby(np.asarray(clusters)).sum()
    G = len(g)
    if G < 2:
        return np.nan
    return float(np.sqrt((g ** 2).sum() / n ** 2 * G / (G - 1)))


def two_sided_p(t, df=None) -> float:
    """Chance of a t-stat at least this far from zero by luck (Student's t; normal if df is None)."""
    if t is None or not np.isfinite(t):
        return np.nan
    if df is None or df <= 0:
        return float(2 * sst.norm.sf(abs(t)))
    return float(2 * sst.t.sf(abs(t), df))


def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini & Hochberg (1995) false-discovery-rate q-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return q


def _adjust(rows, key="p"):
    pv = [r[key] for r in rows]
    ok = [i for i, p in enumerate(pv) if p is not None and np.isfinite(p)]
    hp, bq = np.full(len(pv), np.nan), np.full(len(pv), np.nan)
    if ok:
        hp[ok] = holm([pv[i] for i in ok])
        bq[ok] = benjamini_hochberg([pv[i] for i in ok])
    for r, h, q in zip(rows, hp, bq):
        r["holm_p"] = round(float(h), 3) if np.isfinite(h) else None
        r["bh_q"] = round(float(q), 3) if np.isfinite(q) else None


def edge_stats(t: pd.DataFrame, col: str = "edge_pct") -> dict:
    """Summary of a group of trades: returns, hit rates and the edge over random twins."""
    if t.empty:
        return {"trades": 0}
    t = t.reset_index(drop=True)
    r = t["return_pct"]
    e = t[col].dropna()
    month = t.loc[e.index, "signal_date"].astype(str).str[:7]
    se = clustered_se(e, month)
    G = int(month.nunique())
    se_naive = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan
    if not np.isfinite(se) or se <= 0:
        se, G = se_naive, len(e)
    tstat = e.mean() / se if np.isfinite(se) and se > 0 else np.nan
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": int(len(t)), "win_rate": round(float((r > 0).mean()), 3),
        "avg_return": round(float(r.mean()), 2), "median_return": round(float(r.median()), 2),
        "profit_factor": round(float(gains / losses), 2) if losses > 0 else None,
        "target_hit": round(float((t["exit_reason"] == "target").mean()), 3),
        "stopped": round(float(t["exit_reason"].isin(["stop", "line"]).mean()), 3),
        "avg_days": round(float(t["days_held"].mean()), 1),
        "random_avg_return": round(float(t["random_return_pct"].mean()), 2),
        "edge": round(float(e.mean()), 2) if len(e) else None,
        "edge_lo": round(float(e.mean() - 1.96 * se), 2) if np.isfinite(tstat) else None,
        "edge_hi": round(float(e.mean() + 1.96 * se), 2) if np.isfinite(tstat) else None,
        "edge_t": round(float(tstat), 2) if np.isfinite(tstat) else None,
        "p": round(two_sided_p(tstat, G - 1), 3) if np.isfinite(tstat) else None,
        "beat_random": round(float((e > 0).mean()), 3) if len(e) else None,
        "months": G, "too_few": bool(len(t) < MIN_TRADES),
    }


def halves(t: pd.DataFrame) -> pd.Series:
    """'1st half' / '2nd half' of the period, split at the median signal date."""
    dates = sorted(t["signal_date"])
    cut = dates[len(dates) // 2] if dates else ""
    return pd.Series(np.where(t["signal_date"] < cut, f"1st half (before {cut})", f"2nd half (from {cut})"), index=t.index)


def by_group(t: pd.DataFrame, col: str, adjust: bool = False) -> pd.DataFrame:
    rows = [{"group": str(k), **edge_stats(g)} for k, g in t.groupby(col)]
    rows.sort(key=lambda d: -d["trades"])
    if adjust and rows:
        _adjust(rows)
    return pd.DataFrame(rows)


def clustered_diff(y, flag, clusters):
    """Difference in the mean of y between flag == 1 and flag == 0, with a cluster-robust
    standard error (a regression of y on the flag, errors grouped by cluster).
    Returns (difference, standard error, number of clusters)."""
    y = np.asarray(y, float); x = np.asarray(flag, float); g = np.asarray(clusters)
    keep = np.isfinite(y) & np.isfinite(x)
    y, x, g = y[keep], x[keep], g[keep]
    n = len(y)
    if n < 4 or x.min() == x.max():
        return np.nan, np.nan, 0
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    groups = pd.Series(np.arange(n)).groupby(g).indices
    G = len(groups)
    if G < 2:
        return float(beta[1]), np.nan, G
    meat = np.zeros((2, 2))
    for idx in groups.values():
        sg = X[idx].T @ u[idx]
        meat += np.outer(sg, sg)
    V = G / (G - 1) * (n - 1) / (n - 2) * XtX_inv @ meat @ XtX_inv
    return float(beta[1]), float(np.sqrt(max(V[1, 1], 0))), G


def check_report(t: pd.DataFrame) -> pd.DataFrame:
    """Does passing each textbook check go with a better edge? (see the notes at the top)"""
    from .confirm import CHECKS
    t = t.reset_index(drop=True)
    month = t["signal_date"].astype(str).str[:7]
    half = halves(t)
    rows = []
    for c in CHECKS:
        col = f"chk_{c.key}"
        if col not in t:
            continue
        f = t[col]
        judged = f.notna()
        npass, nfail = int((f == 1).sum()), int((f == 0).sum())
        d, se, G = clustered_diff(t.loc[judged, "edge_pct"], f[judged], month[judged])
        tstat = d / se if np.isfinite(se) and se > 0 else np.nan
        hd = []
        for hname in sorted(half.unique()):
            m = judged & (half == hname)
            hd.append(clustered_diff(t.loc[m, "edge_pct"], f[m], month[m])[0])
        enough = min(npass, nfail) >= MIN_TRADES
        rows.append({
            "check": c.key, "label": c.label, "group": c.group, "judged": int(judged.sum()),
            "passed": npass, "failed": nfail,
            "edge_passed": round(float(t.loc[f == 1, "edge_pct"].mean()), 2) if npass else None,
            "edge_failed": round(float(t.loc[f == 0, "edge_pct"].mean()), 2) if nfail else None,
            "difference": round(d, 2) if np.isfinite(d) else None,
            "diff_lo": round(d - 1.96 * se, 2) if np.isfinite(tstat) else None,
            "diff_hi": round(d + 1.96 * se, 2) if np.isfinite(tstat) else None,
            "diff_1st_half": round(hd[0], 2) if len(hd) > 0 and np.isfinite(hd[0]) else None,
            "diff_2nd_half": round(hd[1], 2) if len(hd) > 1 and np.isfinite(hd[1]) else None,
            # only judged when both sides have enough trades; tiny groups give meaningless p-values
            "p": two_sided_p(tstat, G - 1) if np.isfinite(tstat) and enough else None,
        })
    _adjust(rows)
    for r in rows:
        both = (r["diff_1st_half"] or 0) > 0 and (r["diff_2nd_half"] or 0) > 0
        enough = min(r["passed"], r["failed"]) >= MIN_TRADES
        sig = r["holm_p"] is not None and r["holm_p"] < 0.05
        r["p"] = round(r["p"], 3) if r["p"] is not None else None
        r["verdict"] = ("HELPS" if sig and r["difference"] > 0 and both and enough
                        else "HURTS" if sig and r["difference"] < 0 and enough
                        else "too few" if not enough else "no clear effect")
    return pd.DataFrame(rows)


def variant_report(t: pd.DataFrame) -> pd.DataFrame:
    """Each trade-management variant's edge minus the book plan's edge, trade by trade,
    on the same trades and the same twin days."""
    from .twins import VARIANTS
    t = t.reset_index(drop=True)
    month = t["signal_date"].astype(str).str[:7]
    out = []
    for key, label in VARIANTS.items():
        if key == "book":
            continue
        col = f"{key}_edge_pct"
        if col not in t:
            continue
        m = t[col].notna()
        if m.sum() < 2:
            out.append({"variant": key, "label": label, "trades": int(m.sum())})
            continue
        change = (t.loc[m, col] - t.loc[m, "edge_pct"]).reset_index(drop=True)
        se = clustered_se(change, month[m].to_numpy())
        G = int(month[m].nunique())
        tstat = change.mean() / se if np.isfinite(se) and se > 0 else np.nan
        out.append({"variant": key, "label": label, "trades": int(m.sum()),
                    "changed": int((change.abs() > 1e-9).sum()),
                    "edge_book": round(float(t.loc[m, "edge_pct"].mean()), 2),
                    "edge_variant": round(float(t.loc[m, col].mean()), 2),
                    "change": round(float(change.mean()), 2),
                    "change_lo": round(float(change.mean() - 1.96 * se), 2) if np.isfinite(tstat) else None,
                    "change_hi": round(float(change.mean() + 1.96 * se), 2) if np.isfinite(tstat) else None,
                    "p": round(two_sided_p(tstat, G - 1), 3) if np.isfinite(tstat) else None,
                    "too_few": bool(m.sum() < MIN_TRADES)})
    return pd.DataFrame(out)


def full_report(t: pd.DataFrame) -> dict:
    """Every table of the random-twin test, as DataFrames."""
    if t is None or t.empty:
        return {}
    t = t.reset_index(drop=True)
    t["half"] = halves(t)
    overall = pd.DataFrame([{"group": "All pattern trades", **edge_stats(t)}])
    return {"overall": overall, "by_half": by_group(t, "half"), "by_pattern": by_group(t, "pattern", adjust=True),
            "by_bias": by_group(t, "bias"), "by_exit": by_group(t, "exit_reason"),
            "checks": check_report(t) if any(c.startswith("chk_") for c in t.columns) else pd.DataFrame(),
            "variants": variant_report(t)}
