"""Framework 9 - validation (version 1).

Included now:  step 1 (time split), step 2 (comparison with buy-and-hold,
one-sided Welch t-tests, buy / sell / buy-minus-sell), step 5 (Bernoulli test
across stocks, on the Watch list tab), step 7 (trading costs) and step 8
(hit rates and pattern outcomes).
Planned for version 2: step 3 (bootstrap against random prices), step 4
(patterns on random data), step 6 (data-snooping correction) and the
forecast-accuracy measures.

Returns: a trade is entered at the open of the day after the signal and closed
at the close h days after the signal day:  r = g * ln(Close[t+h] / Open[t+1]).
If an opposite signal from the same rule appears first, the trade closes at the
next day's open. The market benchmark uses the same definition on every day.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from .signals import Signal


def blocks(n: int, s: dict) -> dict:
    a = int(n * float(s["split_train"]))
    b = int(n * (float(s["split_train"]) + float(s["split_valid"])))
    return {"Training": (0, a), "Validation": (a, b), "Final test": (b, n), "Whole history": (0, n)}


def market_returns(open_, close, h):
    n = len(close)
    r = np.full(n, np.nan)
    if n > h + 1:
        t = np.arange(0, n - h)
        ok = t + 1 < n
        r[t[ok]] = np.log(close[t[ok] + h] / open_[t[ok] + 1])
    return r


def rule_trades(sigs: list[Signal], open_, close, h, cost_rt):
    """Return arrays (idx, direction, gross, net) of fixed-period trades."""
    n = len(close)
    sigs = sorted(sigs, key=lambda x: x.idx)
    idxs = np.array([x.idx for x in sigs])
    out = []
    for j, sg in enumerate(sigs):
        g, t = sg.direction, sg.idx
        if g == 0 or t + h >= n or t + 1 >= n:
            continue
        entry = open_[t + 1]
        exit_px = close[t + h]
        for k in range(j + 1, len(sigs)):
            u = sigs[k]
            if u.idx >= t + h:
                break
            if u.idx > t and u.direction != g and u.idx + 1 < n:
                exit_px = open_[u.idx + 1]
                break
        if not (entry > 0 and exit_px > 0):
            continue
        gross = g * math.log(exit_px / entry)
        out.append((t, g, gross, gross - cost_rt))
    if not out:
        return np.array([], int), np.array([], int), np.array([]), np.array([])
    a = np.array(out)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3]


def _welch(x, y):
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    if np.var(x) == 0 and np.var(y) == 0:
        return np.nan, np.nan
    r = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def evaluate_rules(rule_sigs: dict[str, list[Signal]], df: pd.DataFrame, s: dict) -> pd.DataFrame:
    open_ = df["Open"].to_numpy(float); close = df["Close"].to_numpy(float)
    n = len(close)
    cost_rt = 2 * (float(s["cost_one_way"]) + float(s["spread_one_way"]))
    blk = blocks(n, s)
    rows = []
    for h in s["holding_periods"]:
        h = int(h)
        mkt = market_returns(open_, close, h)
        for rule, sigs in rule_sigs.items():
            ti, gi, gross, net = rule_trades(sigs, open_, close, h, cost_rt)
            for bname, (a, b) in blk.items():
                m = (ti >= a) & (ti < b)
                mk = mkt[a:b]; mk = mk[~np.isnan(mk)]
                g = gi[m]; gr = gross[m]; nt = net[m]
                buy = nt[g > 0]; sell = nt[g < 0]
                t_all, p_all = _welch(nt, mk)
                t_buy, p_buy = _welch(buy, mk)
                t_sell, p_sell = _welch(sell, -mk)
                raw_buy = gr[g > 0]; raw_sell = -gr[g < 0]     # raw price moves after buy / sell days
                t_bs, p_bs = _welch(raw_buy, raw_sell)
                rows.append({
                    "rule": rule, "h": h, "block": bname,
                    "n": int(m.sum()), "n_buy": int((g > 0).sum()), "n_sell": int((g < 0).sum()),
                    "mean_net": float(nt.mean()) if len(nt) else np.nan,
                    "mean_buy": float(buy.mean()) if len(buy) else np.nan,
                    "mean_sell": float(sell.mean()) if len(sell) else np.nan,
                    "mean_market": float(mk.mean()) if len(mk) else np.nan,
                    "hit_rate": float((gr > 0).mean()) if len(gr) else np.nan,
                    "t": t_all, "p": p_all, "p_buy": p_buy, "p_sell": p_sell, "p_buy_minus_sell": p_bs,
                })
    return pd.DataFrame(rows)


def pass_table(res: pd.DataFrame, s: dict, min_signals: int = 5) -> pd.DataFrame:
    """Version-1 pass rule: in the FINAL TEST block, at the judging holding
    period, the rule's average return after costs beats buy-and-hold with
    p < alpha (steps 2 and 7). The bootstrap (step 3) is not yet run."""
    h = int(s["judge_period"]); alpha = float(s["alpha"])
    sub = res[(res["block"] == "Final test") & (res["h"] == h)].copy()
    def verdict(r):
        if r["n"] < min_signals:
            return "Too few signals"
        if r["p"] < alpha and r["mean_net"] > r["mean_market"]:
            return "Passed (steps 2 and 7)"
        return "Not passed"
    sub["verdict"] = sub.apply(verdict, axis=1)
    return sub[["rule", "n", "mean_net", "mean_market", "hit_rate", "p", "verdict"]].reset_index(drop=True)


def pattern_outcomes(pattern_sigs: list[Signal], n: int, s: dict) -> pd.DataFrame:
    blk = blocks(n, s)
    rows = []
    by = defaultdict(list)
    for sg in pattern_sigs:
        tr = sg.extra.get("trade") or {}
        if tr.get("status") == "closed":
            by[sg.rule].append((sg.idx, tr["outcome"], tr["ret"]))
    for rule, lst in by.items():
        for bname, (a, b) in blk.items():
            sel = [x for x in lst if a <= x[0] < b]
            if not sel:
                continue
            outc = [x[1] for x in sel]; rets = np.array([x[2] for x in sel])
            rows.append({"rule": rule, "block": bname, "trades": len(sel),
                         "target hit": outc.count("target") / len(sel),
                         "time limit": outc.count("time") / len(sel),
                         "stopped out": outc.count("stop") / len(sel),
                         "avg return": float(rets.mean())})
    return pd.DataFrame(rows)


def bernoulli_test(successes: int, N: int, p0: float = 0.5) -> dict:
    """Framework 9 step 5. Normal approximation for N >= 30, exact binomial otherwise."""
    if N == 0:
        return {"N": 0, "x": 0, "method": "-", "z": np.nan, "p": np.nan}
    if N >= 30:
        z = (successes - p0 * N) / math.sqrt(N * p0 * (1 - p0))
        return {"N": N, "x": successes, "method": "z-test", "z": z, "p": float(1 - stats.norm.cdf(z))}
    p = float(stats.binom.sf(successes - 1, N, p0))
    return {"N": N, "x": successes, "method": "exact binomial", "z": np.nan, "p": p}
