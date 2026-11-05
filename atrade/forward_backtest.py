"""Train the three indicators across many stocks and prove how well they work,
out of sample.

Because true tops and bottoms are rare, the models are **pooled**: the clues and
labels of many stocks are stacked together, one model is fitted on the early
part of history, and it is tested on the later part - history the model never
saw while training. This is a walk-forward test, the honest way to judge a
forecasting rule.

What we report for the tops and bottoms models:

* **base rate** - the share of days that really were tops (or bottoms);
* **hit rate when it fires** - of the days the model flagged (probability above a
  threshold), how many really were tops;
* **lift** - the hit rate divided by the base rate: how many times better than
  guessing the average. Lift comfortably above 1 is the goal.

For the continuation model we report how often "trend will continue" and "trend
will reverse" calls were right.

Run it from the command line to fit the final models on all available history
and save them for the app to use::

    ATRADE_OFFLINE=1 python -m atrade.forward_backtest --save
    ATRADE_OFFLINE=1 python -m atrade.forward_backtest --tickers AAPL MSFT JPM XOM
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import data as D
from . import forward_features as FF
from . import forward_labels as FL
from . import forward_model as M
from . import sectors as S
from .cleaning import clean
from .config import DEFAULTS
from .trend import trend_state


def _universe(limit: int | None = None) -> list[str]:
    from .config import ROOT
    p = ROOT / "options_data" / "universe.txt"
    names = [x.strip().upper() for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    names = [n for n in names if n not in S.SECTOR_ETFS and n not in ("SPY", "QQQ", "IWM")]
    return names[:limit] if limit else names


def build_samples(ticker: str, s: dict, sector_closes: dict, spy: pd.Series,
                  label_kw: dict | None = None):
    """Return a dict of aligned feature tables and labels for one stock, trimmed
    to the days whose labels are known (no peeking). Returns ``None`` when the
    stock cannot be loaded or is too short."""
    label_kw = label_kw or {}
    try:
        raw = D.repo_prices(ticker)
        if raw is None or len(raw) < 600:
            return None
        df, _, _ = clean(raw, s)
    except Exception:
        return None
    if len(df) < 600:
        return None
    close = df["Close"].to_numpy(float)
    state = trend_state(close, s)
    # options data is daily and only recent; include when present
    try:
        gex = D.load_gex(ticker, df.index)
    except Exception:
        gex = None
    try:
        sent = D.load_sentiment(ticker, df.index, int(s.get("history_years", 20)))
    except Exception:
        sent = {}
    asg = S.assign_sector(ticker, df["Close"], sector_closes=sector_closes)
    etf = sector_closes.get(asg["etf"]) if asg["etf"] else None
    feats = FF.build_features(df, ticker, gex=gex, sentiment=sent,
                              sector_etf_close=etf, market_close=spy)

    is_top, is_bot, last_tb = FL.label_tops_bottoms(df, **{k: v for k, v in label_kw.items()
                                                           if k in ("window", "drop_atr", "horizon", "atr_w")})
    cont, last_c = FL.label_continuation(df, state, **{k: v for k, v in label_kw.items()
                                                       if k in ("barrier_atr", "horizon", "atr_w")})
    idx = df.index
    cont_feats = M.continuation_features(feats, state)
    return {
        "ticker": ticker, "index": idx, "feats": feats, "cont_feats": cont_feats,
        "is_top": is_top, "is_bottom": is_bot, "cont": cont,
        "last_tb": last_tb, "last_c": last_c, "n": len(df),
    }


def _stack(samples: list[dict], split_frac: float = 0.7):
    """Stack per-stock tables into pooled train / test matrices. Each stock is
    split at the same fraction of ITS OWN history, so the test set is always the
    later period for every stock."""
    feat_cols = None
    cont_cols = None
    for sm in samples:
        if sm is None:
            continue
        feat_cols = list(sm["feats"].columns)
        cont_cols = list(sm["cont_feats"].columns)
        break
    tr = {"Xtop": [], "ytop": [], "Xbot": [], "ybot": [], "Xcont": [], "ycont": []}
    te = {k: [] for k in tr}
    for sm in samples:
        if sm is None:
            continue
        n = sm["n"]
        cut = int(n * split_frac)
        F = sm["feats"].reindex(columns=feat_cols).to_numpy(float)
        C = sm["cont_feats"].reindex(columns=cont_cols).to_numpy(float)
        for arr_name, y_src, Xsrc, cols in (("top", sm["is_top"], F, feat_cols),
                                            ("bot", sm["is_bottom"], F, feat_cols)):
            y = y_src
            known = np.isfinite(y)
            known[sm["last_tb"] + 1:] = False
            tr_mask = known.copy(); tr_mask[cut:] = False
            te_mask = known.copy(); te_mask[:cut] = False
            tr[f"X{arr_name}"].append(Xsrc[tr_mask]); tr[f"y{arr_name}"].append(y[tr_mask])
            te[f"X{arr_name}"].append(Xsrc[te_mask]); te[f"y{arr_name}"].append(y[te_mask])
        # continuation: resolved days only (continue vs reverse); label 1 = reverse
        cont = sm["cont"]
        resolved = np.isfinite(cont) & (cont != FL.STALL)
        resolved[sm["last_c"] + 1:] = False
        yc = (cont == FL.REVERSE).astype(float)
        tr_mask = resolved.copy(); tr_mask[cut:] = False
        te_mask = resolved.copy(); te_mask[:cut] = False
        tr["Xcont"].append(C[tr_mask]); tr["ycont"].append(yc[tr_mask])
        te["Xcont"].append(C[te_mask]); te["ycont"].append(yc[te_mask])

    def cat(d):
        return {k: (np.concatenate(v) if v and any(len(a) for a in v) else np.empty((0,)))
                for k, v in d.items()}
    return cat(tr), cat(te), feat_cols, cont_cols


def _eval_binary(model, X, y, thresholds=(0.15, 0.25, 0.4)):
    """Hit rate and lift of a yes/no model at a few probability thresholds."""
    if len(y) == 0:
        return {"base_rate": np.nan, "n": 0, "rows": []}
    w = np.r_[model["intercept"], model["coef"]]
    p = M.predict_proba(w, X)
    base = float(y.mean())
    rows = []
    for th in thresholds:
        fired = p >= th
        nf = int(fired.sum())
        hit = float(y[fired].mean()) if nf else np.nan
        rows.append({"threshold": th, "n_fired": nf,
                     "share_fired": nf / len(y), "hit_rate": hit,
                     "lift": (hit / base) if (nf and base > 0) else np.nan})
    # area under ROC via rank statistic (Mann-Whitney)
    auc = _auc(p, y)
    return {"base_rate": base, "n": int(len(y)), "auc": auc, "rows": rows}


def _auc(p, y):
    y = np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(tickers: list[str] | None = None, limit: int | None = 120, split_frac: float = 0.7,
        l2: float = 2.0, label_kw: dict | None = None, save: bool = False, verbose: bool = True):
    """Build samples, pool-train on the early period, test on the later period,
    print the scorecard, and (optionally) refit on ALL history and save the
    models for the app."""
    os.environ.setdefault("ATRADE_OFFLINE", "1")
    s = dict(DEFAULTS)
    tickers = tickers or _universe(limit)
    sector_closes = S.load_sector_closes()
    spy = S._close("SPY")
    if verbose:
        print(f"Loading {len(tickers)} stocks ...")
    samples = []
    for i, t in enumerate(tickers):
        sm = build_samples(t, s, sector_closes, spy, label_kw)
        if sm is not None:
            samples.append(sm)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(tickers)} ({len(samples)} usable)")
    if verbose:
        print(f"{len(samples)} usable stocks.")
    tr, te, feat_cols, cont_cols = _stack(samples, split_frac)

    report = {"n_stocks": len(samples), "feature_cols": feat_cols, "cont_cols": cont_cols}
    models = {}
    for name, Xtr, ytr, Xte, yte, cols in (
        ("top", tr["Xtop"], tr["ytop"], te["Xtop"], te["ytop"], feat_cols),
        ("bottom", tr["Xbot"], tr["ybot"], te["Xbot"], te["ybot"], feat_cols),
        ("continuation", tr["Xcont"], tr["ycont"], te["Xcont"], te["ycont"], cont_cols),
    ):
        if len(ytr) < 200 or ytr.sum() < 20:
            report[name] = {"error": "too few training events"}
            continue
        mdl = M.train_binary(Xtr, ytr, cols, l2=l2)
        ev = _eval_binary(mdl, Xte, yte)
        report[name] = {"train": {"n": mdl["n"], "n_pos": mdl["n_pos"], "base_rate": mdl["base_rate"]},
                        "test": ev, "top_weights": _top_weights(mdl)}
        models[name] = mdl
        if verbose:
            _print_block(name, report[name])

    if save:
        # refit on ALL known history (train+test pooled) for production use
        full = _stack(samples, split_frac=1.0)[0]
        stall_rate = _stall_rate(samples)
        for name, Xk, yk, cols in (("top", "Xtop", "ytop", feat_cols),
                                   ("bottom", "Xbot", "ybot", feat_cols),
                                   ("continuation", "Xcont", "ycont", cont_cols)):
            X, y = full[Xk], full[yk]
            if len(y) >= 200 and y.sum() >= 20:
                mdl = M.train_binary(X, y, cols, l2=l2)
                if name == "continuation":
                    mdl["stall_rate"] = stall_rate
                fname = {"top": M.MODEL_FILE_TOP, "bottom": M.MODEL_FILE_BOTTOM,
                         "continuation": M.MODEL_FILE_CONT}[name]
                M.save_model(mdl, fname)
                if verbose:
                    print(f"saved {fname}  (n={mdl['n']}, events={mdl['n_pos']})")
    return report, models


def _stall_rate(samples) -> float:
    """Share of trending days where neither barrier was hit in time (the trend
    stalled). Stored on the continuation model so the app can split the odds into
    continue / reverse / stall."""
    tot = stall = 0
    for sm in samples:
        if sm is None:
            continue
        cont = sm["cont"]
        ok = np.isfinite(cont)
        ok[sm["last_c"] + 1:] = False
        tot += int(ok.sum())
        stall += int((cont[ok] == FL.STALL).sum())
    return float(stall / tot) if tot else 0.08


def _top_weights(mdl, k=6):
    pairs = sorted(zip(mdl["features"], mdl["coef"]), key=lambda x: -abs(x[1]))
    return [(f, round(c, 3)) for f, c in pairs[:k]]


def _print_block(name, r):
    if "error" in r:
        print(f"\n[{name}] {r['error']}")
        return
    t = r["test"]
    print(f"\n[{name}]  train events={r['train']['n_pos']}/{r['train']['n']} "
          f"(base {r['train']['base_rate']:.1%})   test base {t['base_rate']:.1%}, AUC {t.get('auc', float('nan')):.3f}")
    print("  threshold  fired   hit-rate   lift")
    for row in t["rows"]:
        hr = f"{row['hit_rate']:.1%}" if np.isfinite(row['hit_rate']) else "  -  "
        lf = f"{row['lift']:.2f}x" if np.isfinite(row['lift']) else "  -  "
        print(f"    p>={row['threshold']:.2f}   {row['share_fired']:5.1%}   {hr:>7}   {lf:>6}")
    print("  strongest clues:", ", ".join(f"{f}:{c:+.2f}" for f, c in r["top_weights"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=120, help="max stocks from the universe")
    ap.add_argument("--l2", type=float, default=2.0)
    ap.add_argument("--save", action="store_true", help="refit on all history and save models for the app")
    args = ap.parse_args()
    run(tickers=args.tickers, limit=args.limit, l2=args.l2, save=args.save)


if __name__ == "__main__":
    main()
