"""Phase 3 - individual-stock regime & capitulation bottoms via correlation-to-vol.

Per stock, causal features: own realized-vol spike, trailing drawdown, and the
stock's correlation to market volatility (rolling corr of stock returns with
dVIX). A stock 'capitulation bottom' fires when the stock's vol spikes AND it is
in a deep drawdown AND the market itself is stressed. Evaluated by whether
forward stock returns after a flag beat the unconditional average (pooled across
all stocks), tuned pre-2016 / tested after.
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, pandas as pd
import driver, phase3

OUT = "/home/user/academic_trading/results/sentiment_tuning"; SPLIT = phase3.SPLIT


def main():
    spy = pd.read_parquet("price_data/SPY.parquet")["Close"]; spy.index = pd.to_datetime(spy.index).normalize()
    vix = pd.read_parquet("price_data/_VIX.parquet")["Close"]; vix.index = pd.to_datetime(vix.index).normalize()
    panel = driver.build_panel_series(spy)
    dates = pd.DatetimeIndex(panel.dates); C = panel.C; J = C.shape[1]
    vix = vix.reindex(dates)
    LR = np.vstack([np.full((1, J), np.nan), np.log(C[1:]/C[:-1])])
    Cdf = pd.DataFrame(C, index=dates)
    lr = pd.DataFrame(LR, index=dates)
    rv = lr.rolling(20).std()*np.sqrt(252)                 # per-stock realized vol
    vspike = rv/rv.rolling(50).mean()                      # per-stock vol spike
    dd = Cdf/Cdf.cummax()-1                                 # per-stock trailing drawdown
    mkt_vspike = (vix/vix.rolling(50).mean())              # market vol spike (vector)
    # forward returns
    fwd20 = np.log(Cdf.shift(-20)/Cdf); fwd60 = np.log(Cdf.shift(-60)/Cdf)
    tune = dates < SPLIT; test = dates >= SPLIT

    def evalflag(a, d, m, era_mask):
        flag = (vspike > a) & (dd < d) & (mkt_vspike.values[:,None] > m)
        valid = ~np.isnan(fwd20.values) & np.isfinite(vspike.values) & np.isfinite(dd.values)
        rowmask = era_mask[:,None]
        fl = flag.values & valid & rowmask
        base = valid & rowmask
        if fl.sum() < 50:
            return None
        return dict(a=a,d=d,m=m, n_flags=int(fl.sum()),
                    flag_fwd20=float(np.nanmean(fwd20.values[fl])),
                    base_fwd20=float(np.nanmean(fwd20.values[base])),
                    flag_fwd60=float(np.nanmean(fwd60.values[fl])),
                    base_fwd60=float(np.nanmean(fwd60.values[base])),
                    flag_pos20=float(np.nanmean(fwd20.values[fl]>0)),
                    base_pos20=float(np.nanmean(fwd20.values[base]>0)))

    grid = list(itertools.product([1.3,1.5,1.8],[-0.15,-0.25,-0.35],[1.0,1.1,1.2]))
    best=None
    for a,d,m in grid:
        r=evalflag(a,d,m,tune)
        if r is None: continue
        edge=r["flag_fwd60"]-r["base_fwd60"]
        if best is None or edge>best[0]: best=(edge,a,d,m,r)
    _,a,d,m,rt=best
    rte=evalflag(a,d,m,test)
    print(f"INDIVIDUAL-STOCK capitulation bottom: stock vspike>{a}, dd<{d}, market vspike>{m}")
    print(f"(forward returns after a flag vs the unconditional average, pooled across stocks)\n")
    print(f"{'era':5s} {'flags':>8s} {'flag_fwd20':>11s} {'base_fwd20':>11s} {'flag_fwd60':>11s} {'base_fwd60':>11s} {'flag_pos20':>11s} {'base_pos20':>11s}")
    for era,r in (("tune",rt),("test",rte)):
        if r: print(f"{era:5s} {r['n_flags']:8d} {100*r['flag_fwd20']:10.2f}% {100*r['base_fwd20']:10.2f}% "
                    f"{100*r['flag_fwd60']:10.2f}% {100*r['base_fwd60']:10.2f}% {100*r['flag_pos20']:10.1f}% {100*r['base_pos20']:10.1f}%")
    json.dump(dict(kind="stock_capitulation", rule=dict(stock_vspike_gt=a,dd_lt=d,mkt_vspike_gt=m),
                   tune=rt,test=rte), open(f"{OUT}/phase3_model_stocks.json","w"), indent=2, default=float)
    print(f"\nwrote {OUT}/phase3_model_stocks.json")


if __name__ == "__main__":
    main()
