"""Framework 9, step 3 - the bootstrap test against random prices (book ch. 3 and 8,
following Brock, Lakonishok and LeBaron 1992).

1. Fit three null models ("nothing special is happening") to the stock's daily
   log returns (in percent):
     AR(1)    r_t = b + rho r_{t-1} + e_t                                   (OLS, eq. 8.8)
     GARCH-M  r_t = alpha + gamma h_t + b e_{t-1} + e_t
              h_t = a0 + a1 e_{t-1}^2 + beta h_{t-1},  e_t = sqrt(h_t) z_t     (ML, eqs. 8.9-8.11)
     E-GARCH  r_t = a + gamma r_{t-1} + beta e_{t-1} + e_t
              ln s2_t = kappa + G ln s2_{t-1} + A(|z_{t-1}| - sqrt(2/pi)) + L z_{t-1}  (ML, 8.12-8.14)
2. Keep a model only if it FITS - its key coefficients are significant (tool choice:
   rho for AR(1); a1 and beta for GARCH-M; G and A for E-GARCH, at the 5% level) - AND its
   leftover shocks look independent and evenly sized: Ljung-Box test (no leftover
   correlation) and Engle's ARCH test (no leftover volatility clustering), both at
   20 lags and the 10% level. Otherwise the model is not used; if no model qualifies
   the stock is "not testable".
3. Build B artificial return histories per qualifying model by drawing the (standardized)
   residuals at random with replacement and feeding them back through the fitted model.
4. Run every price-based rule, with the same costs, on every artificial history.
5. p = share of artificial histories whose mean rule return is at least as high as on the
   real prices (p = (1 + count) / (B + 1) is also reported). The share of artificial
   histories with a HIGHER volatility after signals is reported too.

Returns in this test are close-to-close, r = g ln(P_{t+h} / P_t), on the real AND the
artificial prices (artificial histories have no opens), with the same costs on both sides.
Sentiment and GEX rules depend on data that is not simulated, so they are not tested here.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import optimize, stats

LAGS = 20
IID_ALPHA = 0.10
SIG_ALPHA = 0.05


# ------------------------------------------------------------ IID tests -----

def ljung_box(x: np.ndarray, lags: int = LAGS) -> float:
    """p-value of the Ljung-Box Q test (H0: no autocorrelation up to `lags`)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    x = x - x.mean()
    den = (x * x).sum()
    q = 0.0
    for k in range(1, lags + 1):
        rho = (x[k:] * x[:-k]).sum() / den
        q += rho * rho / (n - k)
    q *= n * (n + 2)
    return float(stats.chi2.sf(q, lags))


def arch_test(x: np.ndarray, lags: int = LAGS) -> float:
    """p-value of Engle's ARCH LM test (H0: no ARCH effects): n R^2 of e_t^2 on its lags."""
    e2 = np.asarray(x, float) ** 2
    e2 = e2[np.isfinite(e2)]
    y = e2[lags:]
    X = np.column_stack([np.ones(len(y))] + [e2[lags - k:-k] for k in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return float(stats.chi2.sf(len(y) * r2, lags))


# ---------------------------------------------------------------- models ----

def _hessian_se(f, x, eps=1e-4):
    """Standard errors from a central-difference Hessian of the negative log-likelihood.
    Steps shrink until every probe point stays inside the allowed parameter region."""
    k = len(x); H = np.zeros((k, k))
    h = eps * np.maximum(np.abs(x), 1e-2)
    f0 = f(x)
    for i in range(k):
        for _ in range(12):
            xp = x.copy(); xp[i] += 2 * h[i]; xm = x.copy(); xm[i] -= 2 * h[i]
            if f(xp) < 1e11 and f(xm) < 1e11:
                break
            h[i] /= 4
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        se[~np.isfinite(se) | (np.diag(cov) <= 0)] = np.nan
        return se
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


class AR1:
    name = "AR(1)"
    params = ["b", "rho"]
    key = ["rho"]

    def fit(self, r):
        y, x = r[1:], r[:-1]
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        s2 = res @ res / (len(y) - 2)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        self.coef, self.se = coef, se
        self.resid = res                      # AR(1): raw residuals are resampled
        self.r0 = r[0]
        return self

    def simulate(self, z, rng=None):
        b, rho = self.coef
        out = np.empty(len(z) + 1); out[0] = self.r0
        for t in range(1, len(out)):
            out[t] = b + rho * out[t - 1] + z[t - 1]
        return out


class GARCHM:
    name = "GARCH-M"
    params = ["alpha", "gamma", "b", "a0", "a1", "beta"]
    key = ["a1", "beta"]

    @staticmethod
    def _filter(th, r):
        alpha, gamma, b, a0, a1, beta = th
        n = len(r); h = np.empty(n); e = np.empty(n)
        h_prev = r.var(); e_prev = 0.0
        for t in range(n):
            ht = a0 + a1 * e_prev * e_prev + beta * h_prev
            if not (ht > 1e-12):
                ht = 1e-12
            elif ht > 1e8:
                ht = 1e8
            et = r[t] - alpha - gamma * ht - b * e_prev
            h[t] = ht; e[t] = et
            h_prev, e_prev = ht, et
        return e, h

    def _nll(self, th, r):
        a0, a1, beta = th[3], th[4], th[5]
        if a0 <= 0 or a1 < 0 or beta < 0 or a1 + beta >= 1.0:
            return 1e12
        with np.errstate(all="ignore"):
            e, h = self._filter(th, r)
            v = 0.5 * np.sum(np.log(2 * np.pi * h) + e * e / h)
        return float(v) if np.isfinite(v) else 1e12

    def fit(self, r):
        v = r.var()
        x0 = np.array([r.mean(), 0.0, 0.0, 0.05 * v, 0.08, 0.90])
        bnds = [(None, None), (None, None), (-0.99, 0.99), (1e-8, None), (0, 0.999), (0, 0.999)]
        f = lambda th: self._nll(th, r)
        res = optimize.minimize(f, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 500})
        self.coef = res.x
        self.se = _hessian_se(f, res.x.copy())
        e, h = self._filter(res.x, r)
        self.resid = e / np.sqrt(h)          # standardized residuals are resampled
        self.state0 = (r.var(), 0.0)
        return self

    def simulate(self, z, rng=None):
        alpha, gamma, b, a0, a1, beta = self.coef
        h_prev, e_prev = self.state0
        out = np.empty(len(z))
        for t in range(len(z)):
            ht = max(a0 + a1 * e_prev * e_prev + beta * h_prev, 1e-12)
            et = math.sqrt(ht) * z[t]
            out[t] = alpha + gamma * ht + b * e_prev + et
            h_prev, e_prev = ht, et
        return out


class EGARCH:
    name = "E-GARCH"
    params = ["a", "gamma", "beta", "kappa", "G", "A", "L"]
    key = ["G", "A"]
    C = math.sqrt(2 / math.pi)

    def _filter(self, th, r):
        a, gamma, beta, kappa, G, A, L = th
        n = len(r); ls = np.empty(n); e = np.empty(n)
        ls_prev = math.log(r.var()); z_prev = 0.0; e_prev = 0.0; r_prev = r.mean()
        for t in range(n):
            lst = kappa + G * ls_prev + A * (abs(z_prev) - self.C) + L * z_prev
            lst = min(max(lst, -30.0), 30.0)
            et = r[t] - a - gamma * r_prev - beta * e_prev
            st = math.exp(0.5 * lst)
            ls[t] = lst; e[t] = et
            ls_prev, z_prev, e_prev, r_prev = lst, et / st, et, r[t]
        return e, np.exp(ls)

    def _nll(self, th, r):
        if abs(th[4]) >= 1.0:
            return 1e12
        with np.errstate(all="ignore"):
            e, s2 = self._filter(th, r)
            v = 0.5 * np.sum(np.log(2 * np.pi * s2) + e * e / s2)
        return float(v) if np.isfinite(v) else 1e12

    def fit(self, r):
        v = r.var()
        x0 = np.array([r.mean(), 0.0, 0.0, 0.02 * math.log(v) if v > 0 else 0.0, 0.98, 0.15, -0.05])
        x0[3] = (1 - 0.98) * math.log(v)
        bnds = [(None, None), (-0.99, 0.99), (-0.99, 0.99), (None, None), (-0.999, 0.999), (None, None), (None, None)]
        f = lambda th: self._nll(th, r)
        res = optimize.minimize(f, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 500})
        self.coef = res.x
        self.se = _hessian_se(f, res.x.copy())
        e, s2 = self._filter(res.x, r)
        self.resid = e / np.sqrt(s2)
        self.state0 = (math.log(v), 0.0, 0.0, r.mean())
        return self

    def simulate(self, z, rng=None):
        a, gamma, beta, kappa, G, A, L = self.coef
        ls_prev, z_prev, e_prev, r_prev = self.state0
        out = np.empty(len(z))
        for t in range(len(z)):
            lst = min(max(kappa + G * ls_prev + A * (abs(z_prev) - self.C) + L * z_prev, -30.0), 30.0)
            et = math.exp(0.5 * lst) * z[t]
            out[t] = a + gamma * r_prev + beta * e_prev + et
            ls_prev, z_prev, e_prev, r_prev = lst, z[t], et, out[t]
        return out


MODELS = [AR1, GARCHM, EGARCH]


def fit_models(close: np.ndarray) -> list[dict]:
    """Fit all three null models and decide which ones qualify for the bootstrap."""
    r = 100 * np.diff(np.log(np.asarray(close, float)))
    r = r[np.isfinite(r)]
    out = []
    for M in MODELS:
        m = M()
        try:
            m.fit(r)
        except Exception as e:      # numerical failure: model does not qualify
            out.append({"model": M.name, "qualified": False, "reason": f"fit failed: {e}", "obj": None})
            continue
        tstat = m.coef / m.se
        pvals = 2 * stats.norm.sf(np.abs(tstat))
        key_ok = all(np.isfinite(pvals[M.params.index(k)]) and pvals[M.params.index(k)] < SIG_ALPHA for k in M.key)
        lb, ar = ljung_box(m.resid), arch_test(m.resid)
        iid = lb > IID_ALPHA and ar > IID_ALPHA
        reason = []
        if not key_ok:
            reason.append("key coefficients not significant")
        if lb <= IID_ALPHA:
            reason.append(f"leftover correlation (Ljung-Box p={lb:.3f})")
        if ar <= IID_ALPHA:
            reason.append(f"leftover volatility clustering (ARCH p={ar:.3f})")
        out.append({"model": M.name, "qualified": key_ok and iid, "reason": "; ".join(reason) or "fits; residuals look IID",
                    "coef": dict(zip(M.params, np.round(m.coef, 6))), "t": dict(zip(M.params, np.round(tstat, 2))),
                    "ljung_box_p": lb, "arch_p": ar, "obj": m})
    return out


def simulate_prices(model, close: np.ndarray, B: int, seed: int = 0, stats: dict | None = None):
    """B artificial price histories with the same start price and length as the real one.

    A history whose price reaches zero or infinity is thrown away and drawn again (tool choice).
    This happens when a GARCH model with very persistent volatility keeps drawing the stock's
    worst real shocks: volatility then spirals until the numbers leave the computer's range.
    Such a path is a numerical breakdown, not a price history. The count is put in stats["redrawn"].
    """
    rng = np.random.default_rng(seed)
    resid = model.resid[np.isfinite(model.resid)]
    n = len(close) - 1
    made = redrawn = 0
    while made < B:
        if redrawn > B:                     # model is unusable; stop rather than loop for ever
            break
        z = rng.choice(resid, size=n if not isinstance(model, AR1) else n - 1, replace=True)
        r = model.simulate(z)
        if isinstance(model, AR1):
            r = r if len(r) == n else r[:n]
        lp = np.log(close[0]) + np.concatenate([[0.0], np.cumsum(r[:n] / 100.0)])
        with np.errstate(over="ignore", under="ignore"):
            px = np.exp(lp)
        if not (np.isfinite(px).all() and (px > 0).all()):
            redrawn += 1
            continue
        made += 1
        yield px
    if stats is not None:
        stats["redrawn"] = redrawn
        stats["made"] = made


# ------------------------------------------------------- rules on closes ----

def price_rules(close: np.ndarray, s: dict) -> dict:
    """Every price-based rule computed from closing prices only: {rule: [Signal]}."""
    from .gauges import gauge_signals
    from .patterns import all_patterns
    from .trend import alexander_signals, trend_signals, trend_state
    from .zones import zone_analysis
    c = np.asarray(close, float)
    df = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c})
    st = trend_state(c, s)
    sigs = trend_signals(df, s) + gauge_signals(c, s) + zone_analysis(c, s)[0] + all_patterns(c, s, st)
    sigs += alexander_signals(c, float(s["alexander_x"]))
    out = defaultdict(list)
    for g in sigs:
        out[g.rule].append(g)
    return out


def close_returns(sigs, close, h, cost_rt):
    """(idx, direction, net return) for fixed-period close-to-close trades; an opposite
    signal from the same rule closes the trade early at that day's close."""
    n = len(close)
    sigs = sorted(sigs, key=lambda x: x.idx)
    rows = []
    for j, g in enumerate(sigs):
        t, d = g.idx, g.direction
        if d == 0 or t + h >= n:
            continue
        exit_t = t + h
        for k in range(j + 1, len(sigs)):
            u = sigs[k]
            if u.idx >= t + h:
                break
            if u.idx > t and u.direction != d:
                exit_t = u.idx
                break
        a, b = close[t], close[exit_t]
        if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)):
            continue                        # no valid price on entry or exit day
        rows.append((t, d, d * math.log(b / a) - cost_rt))
    return rows


def block_stats(rows, a, b):
    sel = [x for x in rows if a <= x[0] < b]
    if not sel:
        return {"n": 0, "mean": np.nan, "buy_mean": np.nan, "sell_mean": np.nan, "buy_sd": np.nan}
    r = np.array([x[2] for x in sel]); d = np.array([x[1] for x in sel])
    return {"n": len(sel), "mean": r.mean(), "buy_mean": r[d > 0].mean() if (d > 0).any() else np.nan,
            "sell_mean": r[d < 0].mean() if (d < 0).any() else np.nan,
            "buy_sd": r[d > 0].std() if (d > 0).sum() > 1 else np.nan}


def run(close: np.ndarray, s: dict, B: int = 500, holding=None, seed: int = 0, progress=None) -> dict:
    """The whole step-3 test for one stock. Returns model fits and a results table."""
    from .validation import blocks
    close = np.asarray(close, float)
    holding = holding or [int(s["judge_period"])]
    cost_rt = 2 * (float(s["cost_one_way"]) + float(s["spread_one_way"]))
    fits = fit_models(close)
    blk = blocks(len(close), s)
    real_rules = price_rules(close, s)
    real = {(rule, h, bn): block_stats(close_returns(sg, close, h, cost_rt), a, b)
            for rule, sg in real_rules.items() for h in holding for bn, (a, b) in blk.items()}
    rows = []
    qualified = [f for f in fits if f["qualified"]]
    for f in qualified:
        sims = defaultdict(list)
        st = {}
        for i, px in enumerate(simulate_prices(f["obj"], close, B, seed, st)):
            rules = price_rules(px, s)
            for rule in real_rules:
                for h in holding:
                    rr = close_returns(rules.get(rule, []), px, h, cost_rt)
                    for bn, (a, b) in blk.items():
                        sims[(rule, h, bn)].append(block_stats(rr, a, b))
            if progress:
                progress(f["model"], i + 1, B)
        f["paths_redrawn"] = st.get("redrawn", 0)
        for key, real_st in real.items():
            sim = sims[key]
            means = np.array([x["mean"] for x in sim], float)
            ok = np.isfinite(means)
            cnt = int((means[ok] >= real_st["mean"]).sum()) if np.isfinite(real_st["mean"]) else None
            bsd = np.array([x["buy_sd"] for x in sim], float)
            rows.append({"rule": key[0], "h": key[1], "block": key[2], "model": f["model"], "n_real": real_st["n"],
                         "mean_real": real_st["mean"], "mean_sim_avg": float(np.nanmean(means)) if ok.any() else np.nan,
                         "p": cnt / ok.sum() if (cnt is not None and ok.sum()) else np.nan,
                         "p_adj": (1 + cnt) / (ok.sum() + 1) if cnt is not None else np.nan,
                         "p_vol": float((bsd[np.isfinite(bsd)] > real_st["buy_sd"]).mean())
                         if np.isfinite(real_st["buy_sd"]) and np.isfinite(bsd).any() else np.nan,
                         "B": int(ok.sum())})
    fit_table = pd.DataFrame([{k: v for k, v in f.items() if k != "obj"} for f in fits])
    if "paths_redrawn" not in fit_table:
        fit_table["paths_redrawn"] = np.nan
    return {"fits": fit_table, "results": pd.DataFrame(rows), "testable": bool(qualified), "B": B,
            "rules": sorted(real_rules)}


def verdict_table(boot: dict, s: dict) -> pd.DataFrame:
    """Step-3 verdict per rule in the FINAL TEST block at the judging holding period:
    passes if p < alpha under EVERY qualifying null model (tool choice: strict)."""
    if not boot:
        return pd.DataFrame(columns=["rule", "bootstrap", "p_max"])
    if not boot.get("testable"):
        return pd.DataFrame([{"rule": r, "bootstrap": "not testable", "p_max": np.nan}
                             for r in boot.get("rules", [])])
    r = boot["results"]
    r = r[(r["block"] == "Final test") & (r["h"] == int(s["judge_period"]))]
    out = []
    for rule, g in r.groupby("rule"):
        ps = g["p_adj"].to_numpy(float)
        if not np.isfinite(ps).any():
            v = "no signals"
        elif np.all(ps < float(s["alpha"])):
            v = "passed"
        else:
            v = "not passed"
        out.append({"rule": rule, "bootstrap": v, "p_max": float(np.nanmax(ps)) if np.isfinite(ps).any() else np.nan})
    return pd.DataFrame(out)
