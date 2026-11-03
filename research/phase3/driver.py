"""Build price panels for a sentiment source, run the tuning sweep on the
tune era, and save results. Sources: 'vix' and 'pc' (Cboe equity put/call)."""
from __future__ import annotations
import sys, glob, os, json, time, itertools
sys.path.insert(0, os.path.dirname(__file__))
PROJECT = "/home/user/academic_trading"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
import numpy as np, pandas as pd
from atrade import sentiment, config
import engine

SCR = os.path.dirname(__file__)
s = config.DEFAULTS
LB = int(s["sp2_lookback"])            # 10
ML = float(s["sp2a_max_loss"])         # 0.10  (stop - not tuned)
TG = float(s["sp2a_trail_giveback"])   # 0.50  (stop - not tuned)
TD = int(s["sp2a_trail_delay"])        # 9     (stop - not tuned)
COST = 2 * (float(s["cost_one_way"]) + float(s["spread_one_way"]))   # 0.002 round trip
SPLIT = 0.70                           # earliest 70% tune, last 30% test

GRID = {
    "vix": dict(entry=[3, 5, 8, 10], exit=[0, 2, 3, 5], trig=[1, 3, 5, 10]),
    "pc":  dict(entry=[2, 3, 4, 5],  exit=[0, 1, 2, 3], trig=[1, 3, 5, 10]),
}


def sent_series(source):
    if source == "vix":
        vx = pd.read_parquet("price_data/_VIX.parquet")["Close"]
        return vx
    if source == "pc":
        import requests
        from atrade import data
        r = requests.get(data.CBOE_EQUITY_PC_URL, timeout=30)
        return data.parse_series_csv(r.content)
    raise ValueError(source)


def build_panel(source):
    return build_panel_series(sent_series(source))


def build_panel_series(sr):
    sr = sr.sort_index()
    sr.index = pd.to_datetime(sr.index).normalize()
    sr = sr[~sr.index.duplicated(keep="last")]
    osc = sentiment.oscillator(sr.to_numpy(float), 50, 100)
    dates = sr.index
    files = sorted(glob.glob("price_data/*.parquet"))
    tickers = [os.path.basename(f)[:-8] for f in files]
    tickers = [t for t in tickers if not t.startswith("_")]      # drop _VIX index
    T, J = len(dates), len(tickers)
    O = np.full((T, J), np.nan); H = O.copy(); L = O.copy(); C = O.copy()
    for j, t in enumerate(tickers):
        d = pd.read_parquet(f"price_data/{t}.parquet")
        d.index = pd.to_datetime(d.index).normalize()
        d = d.reindex(dates)
        O[:, j] = d["Open"].to_numpy(float); H[:, j] = d["High"].to_numpy(float)
        L[:, j] = d["Low"].to_numpy(float);  C[:, j] = d["Close"].to_numpy(float)
    panel = engine.Panel(osc, O, H, L, C, tickers, dates)
    return panel


def buyhold(panel, lo, hi):
    """per-stock buy&hold log-return over [lo,hi): first valid open to last valid close."""
    O, C = panel.O, panel.C; J = panel.J
    out = np.full(J, np.nan)
    sub_o = O[lo:hi]; sub_c = C[lo:hi]
    for j in range(J):
        oj = sub_o[:, j]; cj = sub_c[:, j]
        vo = np.where(~np.isnan(oj))[0]; vc = np.where(~np.isnan(cj))[0]
        if len(vo) and len(vc) and vc[-1] >= vo[0]:
            p0 = oj[vo[0]]; p1 = cj[vc[-1]]
            if p0 > 0 and p1 > 0:
                out[j] = np.log(p1 / p0)
    return out


def run_sweep(source):
    t0 = time.time()
    panel = build_panel(source)
    T = panel.T
    split = int(T * SPLIT)
    # warmup: first tradable osc day
    bh_tune = buyhold(panel, 0, split)
    g = GRID[source]
    combos = list(itertools.product(g["entry"], g["exit"], g["trig"], g["trig"]))
    rows = []
    perstock = {}   # keep per-stock ret for every combo (small: J floats each)
    for i, (eX, xX, tl, ts) in enumerate(combos):
        rs, nl, nsh = engine.sweep_combo(panel, eX, xX, tl, ts, LB, ML, TG, TD, COST, 0, split)
        ntr = nl + nsh
        traded = ntr > 0
        has_era = ~np.isnan(bh_tune)
        # edge per stock (log space); stocks with era data only
        edge = rs - bh_tune
        m = has_era
        mean_edge = float(np.nanmean(edge[m]))
        win = float(np.mean((rs[m] > bh_tune[m]))) if m.sum() else np.nan
        # restrict "win among traded" too
        mt = has_era & traded
        win_tr = float(np.mean(rs[mt] > bh_tune[mt])) if mt.sum() else np.nan
        rows.append(dict(entry=eX, exit=xX, trig_long=tl, trig_short=ts,
                         mean_edge=mean_edge, pct_beat=win, pct_beat_traded=win_tr,
                         total_trades=int(ntr.sum()), trades_per_stock=float(ntr[m].mean()),
                         stocks_traded=int(mt.sum()), stocks=int(m.sum()),
                         mean_strat=float(np.nanmean(rs[m])), mean_bh=float(np.nanmean(bh_tune[m]))))
        perstock[(eX, xX, tl, ts)] = rs.astype(np.float32)
        if (i + 1) % 32 == 0:
            print(f"  {source}: {i+1}/{len(combos)} combos  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{SCR}/sweep_{source}.csv", index=False)
    np.savez_compressed(f"{SCR}/perstock_{source}.npz",
                        **{"__tickers__": np.array(panel.tickers),
                           "__split__": np.array([split, T]),
                           "__bh_tune__": bh_tune,
                           **{"_".join(map(str, k)): v for k, v in perstock.items()}})
    print(f"{source}: done {len(combos)} combos, {panel.J} stocks, T={T}, split={split} "
          f"({panel.dates[0].date()}..{panel.dates[split-1].date()} tune | "
          f"{panel.dates[split].date()}..{panel.dates[-1].date()} test) in {time.time()-t0:.0f}s")
    return df


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "vix"
    run_sweep(src)
