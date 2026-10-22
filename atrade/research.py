"""Cross-stock research runner (Framework 9 on all stocks, the [LMW] 9b tests, and the
Framework 7 weight tuning). Uses the price history in price_data/.

    python -m atrade.research calibrate                       # stage 1: LMW width + directions (training data)
    python -m atrade.research stocks --shard 0 --of 20 --B 500 # stage 2: per-stock work (run every shard)
    python -m atrade.research combine                         # stage 3: pooled tests + weight tuning
    python -m atrade.research all --tickers AAPL MSFT SPY --B 100   # everything, one machine

Stage 2 writes per-stock files into work/ (scores and LMW returns are only needed by
stage 3) and results/ (bootstrap and random-pattern results, kept in the repo).
Stage 3 writes results/summary/*.csv and results/tuned_weights.json.

Nothing here looks at the final-test block when choosing anything: the LMW width and
directions and the rating weights are all chosen on the TRAINING block (first 60%).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULTS, ROOT
from .results import RESULTS, load_json, save_json, save_table

WORK = ROOT / "work"


def universe() -> list[str]:
    p = ROOT / "price_data" / "manifest.csv"
    if p.exists():
        m = pd.read_csv(p)
        m = m[(m["days"] >= 750) & (m["error"].isna() | (m["error"].astype(str) == ""))]
        return sorted(m["ticker"].astype(str))
    return []


def exchange_groups() -> dict:
    p = ROOT / "price_data" / "info.csv"
    if not p.exists():
        return {}
    i = pd.read_csv(p)
    return dict(zip(i["ticker"].astype(str), i.get("exchange_group", pd.Series(["Other"] * len(i))).astype(str)))


def prices(ticker: str) -> pd.DataFrame | None:
    from .cleaning import clean
    from .data import repo_prices
    raw = repo_prices(ticker)
    if raw is None or len(raw) < 750:
        return None
    df, _, _ = clean(raw, DEFAULTS)
    return df


# ------------------------------------------------------------- stage 1 -----

def calibrate(tickers: list[str], s: dict = DEFAULTS, sample: int = 120, log=print) -> dict:
    """LMW bandwidth (pattern rate matched to LMW on training data) and directions
    (pooled 9b information tests on training data)."""
    from . import lmw
    from .validation import blocks
    rng = np.random.default_rng(0)
    pick = list(rng.choice(tickers, size=min(sample, len(tickers)), replace=False))
    train = []
    for t in pick:
        df = prices(t)
        if df is not None:
            c = df["Close"].to_numpy(float); a, b = blocks(len(c), s)["Training"]
            train.append(c[a:b])
    cal = lmw.calibrate_bandwidth(train)
    log(f"LMW bandwidth: {cal['h']} days (rate {cal['rates'][cal['h']]:.2f} per stock-year vs LMW {cal['target']:.2f})")
    cond = {k: [] for k in lmw.NAMES}; unc = []
    for t in tickers:
        df = prices(t)
        if df is None:
            continue
        c = df["Close"].to_numpy(float); a, b = blocks(len(c), s)["Training"]
        det = lmw.detect(c, None, cal["h"])
        cd, u = lmw.post_pattern_returns(det, c, a, b)
        for k, v in cd.items():
            cond[k].append(v)
        unc.append(u)
    U = np.concatenate(unc) if unc else np.array([])
    rows = []
    for k in lmw.NAMES:
        r = lmw.info_tests(np.concatenate(cond[k]) if cond[k] else np.array([]), U)
        r["pattern"] = k
        r["informative"] = bool(r["p_decile"] < 0.05 or r["p_ks"] < 0.05) if np.isfinite(r["p_decile"]) else False
        rows.append(r)
    table = pd.DataFrame(rows)
    directions = lmw.learn_directions(table)
    out = {"h": cal["h"], "rates": cal["rates"], "directions": directions, "stocks_for_width": len(train),
           "stocks_for_directions": len(unc), "chosen_on": "training block (first 60%) only"}
    save_json(out, "lmw_settings.json")
    save_table(table, "summary/lmw_9b_training.csv")
    log(f"LMW directions: {directions}")
    return out


# ------------------------------------------------------------- stage 2 -----

def stock_job(ticker: str, B: int, s: dict = DEFAULTS) -> str:
    from . import bootstrap as BS
    from . import lmw
    from .analysis import analyze
    from .results import save_bootstrap
    from .validation import blocks, random_series_patterns
    t0 = time.time()
    df = prices(ticker)
    if df is None:
        return f"{ticker}: skipped (too little history)"
    r = analyze(ticker, dict(s), prices=df[["Open", "High", "Low", "Close", "Volume"]])
    work = WORK / "stocks"; work.mkdir(parents=True, exist_ok=True)
    q = r["results"]; q = q[q["h"] == int(s["judge_period"])]
    q.assign(ticker=ticker).to_csv(work / f"{ticker}_rules.csv", index=False)
    sc = r["rating"][["T", "R", "S"]].copy(); sc["close"] = r["df"]["Close"].to_numpy()
    sc["open"] = r["df"]["Open"].to_numpy()
    sc.to_parquet(work / f"{ticker}_scores.parquet")
    c = r["df"]["Close"].to_numpy(float)
    rows = []
    for bn, (a, b) in blocks(len(c), s).items():
        cd, u = lmw.post_pattern_returns(r["lmw_detections"], c, a, b)
        for k, v in cd.items():
            rows += [{"block": bn, "pattern": k, "x": float(x)} for x in v]
        rows += [{"block": bn, "pattern": "_all", "x": float(x)} for x in u[:: max(1, len(u) // 400)]]
    pd.DataFrame(rows).to_parquet(work / f"{ticker}_lmw.parquet")
    r["variants"].assign(ticker=ticker).to_csv(work / f"{ticker}_variants.csv", index=False)
    pd.DataFrame([{**r["reality_check"], "ticker": ticker}]).to_csv(work / f"{ticker}_rc.csv", index=False)
    if B > 0:
        boot = BS.run(c, s, B=B)
        save_bootstrap(ticker, boot)
    rp = random_series_patterns(c, s)
    save_table(rp.assign(ticker=ticker), f"random_patterns/{ticker.replace('^', '_')}.csv")
    return f"{ticker}: done in {time.time() - t0:.0f}s"


def safe_stock_job(ticker: str, B: int) -> str:
    """stock_job, but a failure on one stock is written to work/errors/ and the run carries on."""
    import traceback
    try:
        return stock_job(ticker, B)
    except Exception as e:
        d = WORK / "errors"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{ticker.replace('^', '_')}.txt").write_text(traceback.format_exc())
        return f"{ticker}: FAILED ({type(e).__name__}: {e}) - skipped, details in work/errors/"


def run_stocks(tickers: list[str], B: int, workers: int, log=print):
    if workers <= 1:
        for t in tickers:
            log(safe_stock_job(t, B))
        return
    with ProcessPoolExecutor(workers) as pool:
        for msg in pool.map(safe_stock_job, tickers, itertools.repeat(B)):
            log(msg)


# ------------------------------------------------------------- stage 3 -----

def _confirm(raw: np.ndarray, k: int) -> np.ndarray:
    """Vectorised 'a rating changes only after k days in a row' (same result as rating.run_rating)."""
    raw = np.asarray(raw)
    change = np.concatenate([[True], raw[1:] != raw[:-1]])
    grp = np.cumsum(change)
    run = np.arange(len(raw)) - np.maximum.accumulate(np.where(change, np.arange(len(raw)), 0)) + 1
    off = pd.Series(np.where(run >= k, raw, np.nan)).ffill().fillna(0).to_numpy()
    return off.astype(int)


def rating_metric(T, R, S, w, cut, veto, k, open_, close, a, b, h, cost_rt) -> tuple[float, int]:
    """Mean net return of rating trades (next open, h days) minus buy-and-hold, days a..b-1."""
    from .rating import rating_signals
    from .validation import market_returns, rule_trades
    score = w[0] * T + w[1] * R + w[2] * S
    G = np.vstack([T, R, S])
    buy = (score >= cut) & ((G > 0).sum(0) >= 2) & (G.min(0) >= -veto)
    sell = (score <= -cut) & ((G < 0).sum(0) >= 2) & (G.max(0) <= veto)
    raw = np.where(buy, 1, np.where(sell, -1, 0))
    rat = _confirm(raw, k)
    sig = rating_signals(rat)
    ti, gi, gross, net = rule_trades(sig, open_, close, h, cost_rt)
    m = (ti >= a) & (ti < b)
    if m.sum() == 0:
        return np.nan, 0
    mk = market_returns(open_, close, h)[a:b]
    return float(net[m].mean() - np.nanmean(mk)), int(m.sum())


def tune_weights(s: dict = DEFAULTS, log=print, step: float = 0.1, cuts=(0.3, 0.4, 0.5)) -> dict:
    """Framework 7: choose weights and cut-off on the TRAINING block of all stocks; check
    them once on the validation block; report the final test as-is."""
    from .validation import blocks
    files = sorted((WORK / "stocks").glob("*_scores.parquet"))
    data = []
    for f in files:
        d = pd.read_parquet(f)
        n = len(d)
        data.append((d["T"].to_numpy(), d["R"].to_numpy(), d["S"].to_numpy(), d["open"].to_numpy(),
                     d["close"].to_numpy(), blocks(n, s)))
    h = int(s["judge_period"]); cost = 2 * (float(s["cost_one_way"]) + float(s["spread_one_way"]))
    grid = [(round(a, 2), round(b, 2), round(1 - a - b, 2)) for a in np.arange(0, 1.0001, step)
            for b in np.arange(0, 1.0001 - a, step)]

    def evaluate(w, cut, block):
        vals, trades = [], 0
        for T, R, S, o, c, blk in data:
            a, b = blk[block]
            v, nt = rating_metric(T, R, S, w, cut, float(s["veto_level"]), int(s["confirm_days"]), o, c, a, b, h, cost)
            if nt >= 3 and np.isfinite(v):
                vals.append(v); trades += nt
        return (float(np.mean(vals)) if vals else np.nan), len(vals), trades

    log(f"Tuning {len(grid) * len(cuts)} weight/cut-off combinations on {len(data)} stocks (training block) ...")
    best = None
    table = []
    for w in grid:
        for cut in cuts:
            v, ns, nt = evaluate(w, cut, "Training")
            table.append({"w_trend": w[0], "w_reversal": w[1], "w_sentiment": w[2], "cutoff": cut,
                          "train_excess": v, "stocks": ns, "trades": nt})
            if np.isfinite(v) and ns >= max(3, len(data) // 4) and (best is None or v > best[0]):
                best = (v, w, cut)
    save_table(pd.DataFrame(table), "summary/weight_tuning_grid.csv")
    framework = ((float(s["w_trend"]), float(s["w_reversal"]), float(s["w_sentiment"])), float(s["buy_cutoff"]))
    out = {"chosen_on": "training block (first 60%) of all stocks", "holding_period": h,
           "objective": "mean (rating trade return after costs - buy-and-hold), averaged over stocks"}
    if best is None:
        out["tuned"] = None
    else:
        out["tuned"] = {"w_trend": best[1][0], "w_reversal": best[1][1], "w_sentiment": best[1][2], "cutoff": best[2]}
        for label, (w, cut) in (("tuned", (best[1], best[2])), ("framework", framework)):
            for blk in ("Training", "Validation", "Final test"):
                v, ns, nt = evaluate(w, cut, blk)
                out[f"{label}_{blk.lower().replace(' ', '_')}"] = {"excess": v, "stocks": ns, "trades": nt}
    save_json(out, "tuned_weights.json")
    log(f"Tuned weights: {out.get('tuned')}")
    return out


def bernoulli_tables(rules: pd.DataFrame, groups: dict) -> pd.DataFrame:
    from .validation import bernoulli_test
    q = rules[(rules["block"] == "Final test") & (rules["n"] >= 1)].copy()
    q["success"] = q["mean_net"] > q["mean_market"]
    q["group"] = q["ticker"].map(groups).fillna("Other") if groups else "All stocks"
    out = []
    parts = list(q.groupby(["rule", "group"]))
    if groups:
        parts += [((r, "All stocks"), g) for r, g in q.groupby("rule")]
    for (rule, grp), g in parts:
        bt = bernoulli_test(int(g["success"].sum()), len(g))
        out.append({"rule": rule, "stocks": grp, "N": bt["N"], "beat_buy_and_hold": bt["x"], "method": bt["method"],
                    "z": bt["z"], "p": bt["p"], "avg_mean_net": g["mean_net"].mean(), "avg_market": g["mean_market"].mean()})
    return pd.DataFrame(out).sort_values(["stocks", "p"])


def combine(s: dict = DEFAULTS, log=print):
    from . import lmw
    from .bootstrap import verdict_table
    from .results import load_bootstrap
    from .validation import full_pass_table
    work = WORK / "stocks"
    errs = sorted((WORK / "errors").glob("*.txt")) if (WORK / "errors").exists() else []
    save_table(pd.DataFrame([{"ticker": f.stem.replace("_", "^", 1) if f.stem.startswith("_") else f.stem,
                              "error": f.read_text().strip().splitlines()[-1]} for f in errs],
                            columns=["ticker", "error"]), "summary/failed_stocks.csv")
    if errs:
        log(f"{len(errs)} stock(s) failed and were left out: {', '.join(f.stem for f in errs)}")
    rules = pd.concat([pd.read_csv(f) for f in sorted(work.glob("*_rules.csv"))], ignore_index=True)
    groups = exchange_groups()
    save_table(bernoulli_tables(rules, groups), "summary/bernoulli_by_rule.csv")
    # full pass rule per stock, then counts per rule
    passes = []
    for t, g in rules.groupby("ticker"):
        boot = load_bootstrap(t)
        p = full_pass_table(g, s, verdict_table(boot, s) if boot else None)
        passes.append(p.assign(ticker=t))
    P = pd.concat(passes, ignore_index=True)
    save_table(P.groupby(["rule", "verdict"]).size().unstack(fill_value=0).reset_index(), "summary/pass_rule_counts.csv")
    # bootstrap overview
    bt = []
    for f in sorted((RESULTS / "bootstrap").glob("*.json")):
        d = json.loads(f.read_text())
        bt.append({"ticker": d["ticker"], "testable": d["testable"],
                   **{f"{x['model']}_qualified": x["qualified"] for x in d["fits"]},
                   **{f"{x['model']}_redrawn": x.get("paths_redrawn") for x in d["fits"] if x["qualified"]}})
    if bt:
        save_table(pd.DataFrame(bt), "summary/bootstrap_models.csv")
    # LMW 9b pooled, final test, by exchange group
    rows = []
    lmw_files = sorted(work.glob("*_lmw.parquet"))
    L = pd.concat([pd.read_parquet(f).assign(ticker=f.name[:-12]) for f in lmw_files], ignore_index=True) if lmw_files else pd.DataFrame()
    if len(L):
        L["group"] = L["ticker"].map(groups).fillna("All") if groups else "All"
        for blk in ("Training", "Final test"):
            for grp in (sorted(set(L["group"])) if groups else []) + ["All stocks"]:
                sub = L[(L["block"] == blk) & ((L["group"] == grp) | (grp == "All stocks"))]
                U = sub.loc[sub["pattern"] == "_all", "x"].to_numpy()
                for k in lmw.NAMES:
                    r = lmw.info_tests(sub.loc[sub["pattern"] == k, "x"].to_numpy(), U)
                    rows.append({"block": blk, "stocks": grp, "pattern": k, **r,
                                 "informative": bool(r["p_decile"] < 0.05 or r["p_ks"] < 0.05) if np.isfinite(r["p_decile"]) else None})
        save_table(pd.DataFrame(rows), "summary/lmw_9b_pooled.csv")
    # head-to-head: textbook vs LMW, and rating variants
    pairs = [("Head-and-shoulders top", "[LMW] HS-LMW"), ("Inverse head-and-shoulders", "[LMW] IHS-LMW"),
             ("Double top", "[LMW] DTOP-LMW"), ("Double bottom", "[LMW] DBOT-LMW"),
             ("Official rating (variant A)", "[LMW] Shadow rating B"), ("Official rating (variant A)", "[LMW] Shadow rating C")]
    ft = rules[rules["block"] == "Final test"]
    hh = []
    for a_, b_ in pairs:
        A = ft[ft["rule"] == a_]; Bq = ft[ft["rule"] == b_]
        hh.append({"textbook": a_, "lmw": b_, "textbook_trades": int(A["n"].sum()), "lmw_trades": int(Bq["n"].sum()),
                   "textbook_avg_net": float((A["mean_net"] * A["n"]).sum() / A["n"].sum()) if A["n"].sum() else np.nan,
                   "lmw_avg_net": float((Bq["mean_net"] * Bq["n"]).sum() / Bq["n"].sum()) if Bq["n"].sum() else np.nan,
                   "textbook_hit": float((A["hit_rate"] * A["n"]).sum() / A["n"].sum()) if A["n"].sum() else np.nan,
                   "lmw_hit": float((Bq["hit_rate"] * Bq["n"]).sum() / Bq["n"].sum()) if Bq["n"].sum() else np.nan})
    H = pd.DataFrame(hh)
    H["kept"] = np.where(H["lmw_avg_net"] > H["textbook_avg_net"], "LMW item kept (beats textbook)", "LMW item dropped")
    save_table(H, "summary/lmw_head_to_head.csv")
    var = pd.concat([pd.read_csv(f) for f in sorted(work.glob("*_variants.csv"))], ignore_index=True)
    save_table(var.groupby("variant")[["total_return", "buy_and_hold", "excess", "max_drawdown", "hit_rate_days"]]
               .mean().reset_index(), "summary/rating_variants_final_test.csv")
    rc = pd.concat([pd.read_csv(f) for f in sorted(work.glob("*_rc.csv"))], ignore_index=True)
    save_table(rc, "summary/reality_check_by_stock.csv")
    rp = [pd.read_csv(f) for f in sorted((RESULTS / "random_patterns").glob("*.csv"))]
    if rp:
        R_ = pd.concat(rp, ignore_index=True)
        agg = R_.groupby("pattern").agg(real=("real_count", "sum"), random_avg=("random_count_avg", "sum"),
                                        real_target_rate=("real_target_rate", "mean"),
                                        random_target_rate=("random_target_rate", "mean"),
                                        real_avg_return=("real_avg_return", "mean"),
                                        random_avg_return=("random_avg_return", "mean")).reset_index()
        save_table(agg, "summary/random_patterns_all_stocks.csv")
    tune_weights(s, log)
    save_json({"stocks": int(rules["ticker"].nunique()), "created": pd.Timestamp.now().isoformat(timespec="minutes"),
               "exchange_split": bool(groups)}, "summary/run_info.json")
    log(f"Combined results for {rules['ticker'].nunique()} stocks written to results/summary/.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calibrate", "stocks", "combine", "all"])
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    import os
    os.environ["ATRADE_OFFLINE"] = "1"          # use price_data/ (incl. ^VIX) only: same data on every run
    tickers = [t.upper() for t in a.tickers] if a.tickers else universe()
    if a.stage in ("calibrate", "all"):
        calibrate(tickers)
    if a.stage in ("stocks", "all"):
        mine = tickers[a.shard::a.of]
        print(f"Shard {a.shard}/{a.of}: {len(mine)} stocks, B={a.B}")
        run_stocks(mine, a.B, a.workers)
    if a.stage in ("combine", "all"):
        combine()


if __name__ == "__main__":
    sys.exit(main())
