"""Phase 3 - per-stock bottom DETECTION (coincidence with each stock's own lows).

Does the shared capitulation flag (stock's own vol-spike + deep drawdown + market
stressed) land on each stock's own major lows? Ground truth = ZigZag(20%) troughs
per stock. Pooled recall / precision / lead-lag across all stocks, tuned on
troughs before 2016, tested after. One shared rule across stocks (not per-stock).
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, pandas as pd
import driver, phase3

OUT = "/home/user/academic_trading/results/sentiment_tuning"; SPLIT = phase3.SPLIT
WIN = 25; PCT = 0.20     # stocks swing more than the index -> 20% major swings


def clusters(idxs):
    if len(idxs) == 0: return []
    out = []; s = p = idxs[0]
    for i in idxs[1:]:
        if i - p > 5: out.append((s, p)); s = i
        p = i
    out.append((s, p)); return out


def main():
    spy = pd.read_parquet("price_data/SPY.parquet")["Close"]; spy.index = pd.to_datetime(spy.index).normalize()
    vix = pd.read_parquet("price_data/_VIX.parquet")["Close"]; vix.index = pd.to_datetime(vix.index).normalize()
    panel = driver.build_panel_series(spy)
    dates = pd.DatetimeIndex(panel.dates); C = panel.C; J = C.shape[1]
    vix = vix.reindex(dates)
    lr = pd.DataFrame(np.vstack([np.full((1, J), np.nan), np.log(C[1:]/C[:-1])]), index=dates)
    rv = lr.rolling(20).std()*np.sqrt(252)
    vspike = (rv/rv.rolling(50).mean()).to_numpy()
    Cdf = pd.DataFrame(C, index=dates); dd = (Cdf/Cdf.cummax()-1).to_numpy()
    mkt_vspike = (vix/vix.rolling(50).mean()).to_numpy()
    splitpos = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))

    # precompute per-stock troughs (positions), split by era
    troughs = {}
    for j in range(J):
        s = Cdf.iloc[:, j].dropna()
        if len(s) < 300: continue
        _, tr = phase3.zigzag(s, PCT)
        pos = [int(np.where(dates == d)[0][0]) for d in tr if d in dates]
        troughs[j] = pos

    def evaluate(a, d, m, era):
        flagmat = (vspike > a) & (dd < d) & (mkt_vspike[:, None] > m)
        hit = nt = goodcl = ncl = 0; leads = []
        for j, tr in troughs.items():
            trs = [t for t in tr if (t < splitpos) == (era == "tune")]
            if not trs: continue
            idxs = np.where(flagmat[:, j])[0]
            idxs = idxs[(idxs < splitpos) == (era == "tune")] if era == "tune" else idxs[idxs >= splitpos]
            cl = clusters(list(idxs))
            nt += len(trs); ncl += len(cl)
            for t in trs:
                near = [c for c in cl if (c[0]-WIN) <= t <= (c[1]+WIN)]
                if near:
                    hit += 1
                    centres = [(c[0]+c[1])//2 for c in near]
                    leads.append(min(centres, key=lambda x: abs(x-t)) - t)
            for c in cl:
                if any((c[0]-WIN) <= t <= (c[1]+WIN) for t in trs): goodcl += 1
        return dict(a=a, d=d, m=m, recall=hit/max(nt,1), precision=goodcl/max(ncl,1),
                    n_turns=nt, n_alerts=ncl, median_leadlag=float(np.median(leads)) if leads else np.nan)

    best = None
    for a, d, m in itertools.product([1.4,1.7,2.0],[-0.20,-0.30,-0.40],[1.0,1.1]):
        r = evaluate(a, d, m, "tune")
        f = r["recall"]*r["precision"]
        if best is None or f > best[0]: best = (f, a, d, m)
    _, a, d, m = best
    rt, rte = evaluate(a, d, m, "tune"), evaluate(a, d, m, "test")
    print(f"PER-STOCK BOTTOM DETECTION (ZigZag {int(PCT*100)}% troughs, +/-{WIN}d window)")
    print(f"shared rule: stock vspike>{a}, drawdown<{d}, market vspike>{m}   tune<{SPLIT.date()}<=test\n")
    print(f"{'era':5s} {'turns':>6s} {'alerts':>7s} {'recall':>7s} {'precision':>9s} {'median lead/lag':>15s}")
    for era, r in (("tune", rt), ("test", rte)):
        print(f"{era:5s} {r['n_turns']:6d} {r['n_alerts']:7d} {r['recall']:6.0%} {r['precision']:8.0%} {r['median_leadlag']:+14.0f}d")
    json.dump(dict(kind="stock_bottom_detection", zigzag_pct=PCT, window_days=WIN,
                   rule=dict(stock_vspike_gt=a, dd_lt=d, mkt_vspike_gt=m), tune=rt, test=rte),
              open(f"{OUT}/phase3_model_stock_turns.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/phase3_model_stock_turns.json")


if __name__ == "__main__":
    main()
