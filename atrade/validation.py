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


# ======================================================================
# Version 2 additions
# ======================================================================

# ---- Step 2 extras: non-parametric test and simple-return check -------------

def robustness_checks(rule_sigs: dict, df: pd.DataFrame, s: dict, block: str = "Final test") -> pd.DataFrame:
    """For the judging holding period: Jarque-Bera normality check on the rule's returns,
    a one-sided Mann-Whitney test (no bell-curve assumption) against buy-and-hold, and
    the Welch test repeated on SIMPLE returns (the book: findings should not change)."""
    open_ = df["Open"].to_numpy(float); close = df["Close"].to_numpy(float)
    n = len(close); h = int(s["judge_period"])
    cost_rt = 2 * (float(s["cost_one_way"]) + float(s["spread_one_way"]))
    a, b = blocks(n, s)[block]
    mkt = market_returns(open_, close, h)[a:b]; mkt = mkt[~np.isnan(mkt)]
    rows = []
    for rule, sigs in rule_sigs.items():
        ti, gi, gross, net = rule_trades(sigs, open_, close, h, cost_rt)
        m = (ti >= a) & (ti < b)
        x = net[m]
        if len(x) < 5 or len(mkt) < 5:
            rows.append({"rule": rule, "n": int(len(x))}); continue
        jb = stats.jarque_bera(x).pvalue if len(x) >= 8 else np.nan
        mw = stats.mannwhitneyu(x, mkt, alternative="greater").pvalue
        simple = gi[m] * (np.exp(gross[m] * gi[m]) - 1) - cost_rt
        _, p_simple = _welch(simple, np.exp(mkt) - 1)
        rows.append({"rule": rule, "n": int(len(x)), "normal_p": jb, "p_mannwhitney": mw, "p_simple": p_simple,
                     "normal": bool(jb > 0.05) if np.isfinite(jb) else None})
    return pd.DataFrame(rows)


# ---- Step 4: patterns on random price series -----------------------------------

def random_series_patterns(close: np.ndarray, s: dict, n_series: int = 10, seed: int = 1) -> pd.DataFrame:
    """Count every chart pattern - and how its trades ended - on the real prices and on
    `n_series` random (geometric Brownian motion) series with the same start price,
    length, drift and volatility (book ch. 5, Table 5.4). A pattern is only useful if it
    does clearly better on the real prices."""
    from .patterns import all_patterns
    from .trades import pattern_trade
    close = np.asarray(close, float)
    lr = np.diff(np.log(close)); mu, sd = lr.mean(), lr.std()
    rng = np.random.default_rng(seed)

    def tally(px):
        sigs = all_patterns(px, s, np.zeros(len(px)))
        out = defaultdict(lambda: {"count": 0, "target": 0, "time": 0, "stop": 0, "ret": []})
        for g in sigs:
            tr = pattern_trade(g, px, px, s)
            o = out[g.rule]; o["count"] += 1
            if tr and tr.get("status") == "closed":
                o[tr["outcome"]] += 1; o["ret"].append(tr["ret"])
        return out

    real = tally(close)
    sims = [tally(close[0] * np.exp(np.concatenate([[0], np.cumsum(rng.normal(mu, sd, len(close) - 1))])))
            for _ in range(n_series)]
    rules = sorted(set(real) | {r for sm in sims for r in sm})
    rows = []
    for r in rules:
        rc = real.get(r, {"count": 0, "target": 0, "time": 0, "stop": 0, "ret": []})
        sc = [sm.get(r, {"count": 0, "target": 0, "time": 0, "stop": 0, "ret": []}) for sm in sims]
        s_cnt = sum(x["count"] for x in sc); s_tgt = sum(x["target"] for x in sc)
        s_closed = sum(x["target"] + x["time"] + x["stop"] for x in sc)
        r_closed = rc["target"] + rc["time"] + rc["stop"]
        # z-test: is the share of target hits higher on real prices than on random ones?
        p1 = rc["target"] / r_closed if r_closed else np.nan
        p2 = s_tgt / s_closed if s_closed else np.nan
        if r_closed and s_closed:
            pp = (rc["target"] + s_tgt) / (r_closed + s_closed)
            se = math.sqrt(pp * (1 - pp) * (1 / r_closed + 1 / s_closed)) if 0 < pp < 1 else np.nan
            z = (p1 - p2) / se if se and np.isfinite(se) else np.nan
            pz = float(stats.norm.sf(z)) if np.isfinite(z) else np.nan
        else:
            z = pz = np.nan
        rr = np.array(rc["ret"]); sr = np.array([v for x in sc for v in x["ret"]])
        rows.append({"pattern": r, "real_count": rc["count"], "random_count_avg": s_cnt / n_series,
                     "real_target_rate": p1, "random_target_rate": p2, "z": z, "p": pz,
                     "real_avg_return": rr.mean() if len(rr) else np.nan,
                     "random_avg_return": sr.mean() if len(sr) else np.nan})
    return pd.DataFrame(rows)


# ---- Step 6: data-snooping correction ---------------------------------------------

def rule_daily_excess(rule_sigs: dict, close: np.ndarray, h: int, cost_rt: float, a: int, b: int) -> pd.DataFrame:
    """Daily excess return over buy-and-hold for each rule: a signal sets the position
    (+1 / -1) for h days (an opposite signal switches it); costs are taken when the
    position changes. Close-to-close, days a..b-1."""
    n = len(close)
    r = np.zeros(n); r[1:] = np.diff(np.log(close))
    cols = {}
    for rule, sigs in rule_sigs.items():
        pos = np.zeros(n)
        for g in sorted(sigs, key=lambda x: x.idx):
            if g.direction == 0:
                continue
            pos[g.idx + 1:min(n, g.idx + 1 + h)] = g.direction
        strat = pos * r - cost_rt / 2 * np.abs(np.diff(np.concatenate([[0], pos])))
        cols[rule] = (strat - r)[a:b]
    return pd.DataFrame(cols)


def _stationary_bootstrap_idx(n, B, q, rng):
    """Politis-Romano stationary bootstrap: random blocks with mean length 1/q."""
    idx = np.empty((B, n), dtype=int)
    for b in range(B):
        i = rng.integers(n)
        for t in range(n):
            if t == 0:
                idx[b, t] = i
            elif rng.random() < q:
                idx[b, t] = rng.integers(n)
            else:
                idx[b, t] = (idx[b, t - 1] + 1) % n
    return idx


def reality_check(excess: pd.DataFrame, B: int = 1000, q: float = 0.1, seed: int = 7) -> dict:
    """White's Reality Check and Hansen's SPA test (consistent version).

    H0: the BEST of all the rules tried has no real edge over the benchmark. A small
    p-value means at least one rule beats buy-and-hold even after allowing for the
    number of rules tested ('data snooping')."""
    X = excess.dropna(axis=1, how="all").fillna(0.0).to_numpy(float)
    if X.size == 0 or X.shape[0] < 50:
        return {"n_rules": X.shape[1] if X.ndim == 2 else 0, "rc_p": np.nan, "spa_p": np.nan, "best_rule": None}
    n, k = X.shape
    fbar = X.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_idx(n, B, q, rng)
    fstar = np.stack([X[i].mean(axis=0) for i in idx])             # B x k
    # White's Reality Check
    V = np.sqrt(n) * fbar.max()
    Vstar = np.sqrt(n) * (fstar - fbar).max(axis=1)
    rc_p = float((Vstar >= V).mean())
    # Hansen's SPA (studentized, recentred)
    omega = np.sqrt(n) * fstar.std(axis=0)
    omega[omega <= 0] = np.nan
    tstat = np.sqrt(n) * fbar / omega
    T = max(np.nanmax(tstat), 0.0)
    thresh = -math.sqrt(2 * math.log(math.log(n)))
    g = np.where(tstat >= thresh, fbar, 0.0)
    Tstar = np.maximum(np.nanmax(np.sqrt(n) * (fstar - g) / omega, axis=1), 0.0)
    spa_p = float((Tstar >= T).mean())
    best = excess.columns[int(np.argmax(fbar))]
    return {"n_rules": k, "days": n, "best_rule": best, "best_mean_excess": float(fbar.max()),
            "rc_p": rc_p, "spa_p": spa_p}


# ---- Step 8: forecast accuracy of the pattern price targets -----------------------

def forecast_accuracy(pattern_sigs: list, close: np.ndarray, start: int = 0, end: int | None = None) -> pd.DataFrame:
    """Treat each pattern's price target as a forecast of the price on its time-limit day
    (book ch. 3.3, eqs. 3.6-3.20). Prices are divided by the break-day close so stocks and
    years can be pooled (x_j = 1). One row per pattern type plus 'All patterns'."""
    end = len(close) if end is None else end
    recs = []
    for g in pattern_sigs:
        if g.target is None or g.time_limit is None or not (start <= g.idx < end) or g.time_limit >= len(close):
            continue
        x0 = close[g.idx]
        recs.append((g.rule, g.idx, 1.0, g.target / x0, close[g.time_limit] / x0))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["rule", "idx", "x", "fc", "act"]).sort_values("idx")

    def measures(d):
        x, f, y = d["x"].to_numpy(), d["fc"].to_numpy(), d["act"].to_numpy()
        e = y - f; m = len(d)
        mse = float(np.mean(e ** 2)); rmse = math.sqrt(mse)
        nr = math.sqrt(np.mean((y.mean() - y) ** 2)); npr = math.sqrt(np.mean((x - y) ** 2))
        prev = np.concatenate([[np.nan], f[:-1]])                      # previous forecast, same pattern type
        ipo = (y - x) * (f - prev)
        out = {"forecasts": m, "MSE": mse, "RMSE": rmse, "MAE": float(np.mean(np.abs(e))),
               "MAPE %": float(100 * np.mean(np.abs(e / y))),
               "NRMSE": rmse / nr if nr > 0 else np.nan, "NPRMSE": rmse / npr if npr > 0 else np.nan,
               "U1": rmse / (math.sqrt(np.mean(f ** 2)) + math.sqrt(np.mean(y ** 2))),
               "U2": math.sqrt(np.sum(((f - y) / x) ** 2) / np.sum(((y - x) / x) ** 2)) if np.sum((y - x) ** 2) > 0 else np.nan,
               "POCID %": float(100 * np.mean((y - x) * (f - x) > 0)),
               "IPOCID %": float(100 * np.mean(ipo[1:] > 0)) if m > 1 else np.nan,
               "POS %": float(100 * np.mean((y - 1) * (f - 1) > 0))}
        if m >= 3 and np.std(f) > 0:                                  # regression of actual on forecast (returns)
            lr = stats.linregress(f - 1, y - 1)
            out.update({"b0": lr.intercept, "b1": lr.slope, "t_b1": lr.slope / lr.stderr if lr.stderr else np.nan,
                        "p_b1": lr.pvalue})
        return out
    rows = [{"pattern": r, **measures(d)} for r, d in df.groupby("rule")]
    rows.append({"pattern": "All patterns", **measures(df)})
    return pd.DataFrame(rows)


# ---- Pass rule with all three steps -------------------------------------------------

def full_pass_table(res: pd.DataFrame, s: dict, boot_verdicts: pd.DataFrame | None, min_signals: int = 5) -> pd.DataFrame:
    """Framework 9 pass rule: in the FINAL TEST block a rule must pass step 2 (beats
    buy-and-hold, one-sided Welch p < alpha), step 3 (bootstrap: beats the random-price
    histories under every qualifying null model) and step 7 (after costs)."""
    t = pass_table(res, s, min_signals)
    t = t.rename(columns={"verdict": "step 2 + 7"})
    if boot_verdicts is not None and len(boot_verdicts):
        t = t.merge(boot_verdicts[["rule", "bootstrap", "p_max"]], on="rule", how="left")
    else:
        t["bootstrap"] = None; t["p_max"] = np.nan

    ran = boot_verdicts is not None and len(boot_verdicts) > 0

    def final(r):
        if r["step 2 + 7"] == "Too few signals":
            return "Too few signals"
        b = r["bootstrap"]
        if b is None or (isinstance(b, float) and np.isnan(b)):
            if not r["step 2 + 7"].startswith("Passed"):
                return "Not passed"
            return "Passed steps 2 + 7 (step 3 not applicable)" if ran else "Bootstrap not run"
        if b == "not testable":
            return "Not testable (no null model fits)"
        return "Passed" if (r["step 2 + 7"].startswith("Passed") and b == "passed") else "Not passed"
    t["verdict"] = t.apply(final, axis=1)
    return t


# ---- Rating variants: performance incl. drawdown (framework 9b head-to-head) ---------

def rating_performance(rating: np.ndarray, close: np.ndarray, a: int, b: int, cost_one_way: float) -> dict:
    """Follow a rating (+1 buy / -1 sell / 0 hold) day by day, close to close, with costs on
    every change. Returns total log return, buy-and-hold, excess, max drawdown, hit rate."""
    r = np.zeros(len(close)); r[1:] = np.diff(np.log(close))
    pos = np.zeros(len(close)); pos[1:] = rating[:-1]          # rating known at the close, held next day
    strat = pos * r - cost_one_way * np.abs(np.diff(np.concatenate([[0], pos])))
    s_, m_ = strat[a:b], r[a:b]
    eq = np.cumsum(s_)
    dd = float((eq - np.maximum.accumulate(np.concatenate([[0], eq]))[1:]).min()) if len(eq) else np.nan
    active = pos[a:b] != 0
    hits = (np.sign(pos[a:b]) * m_)[active]
    return {"total_return": float(s_.sum()), "buy_and_hold": float(m_.sum()), "excess": float(s_.sum() - m_.sum()),
            "max_drawdown": dd, "days_in_market": int(active.sum()),
            "hit_rate_days": float((hits > 0).mean()) if len(hits) else np.nan}
