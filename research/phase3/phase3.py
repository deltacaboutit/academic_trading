"""Phase 3 - a composite 'correlation to volatility' model for regimes and turns.

Distinct from Phase 1 (which used VIX vs its moving average as a cash overlay).
Phase 3's core indicator is the CORRELATION dimension of the market:
  CORR  = realized correlation proxy = Var(index returns) / mean_i Var(stock_i)
          (rises when stocks move together -> systemic stress)
combined with volatility and price context:
  VOL   = VIX level and VIX vs its moving average
  DD    = trailing drawdown from the 252-day peak
  EXT   = price vs its 200-day average (extension / complacency)

All features are causal (use data up to day t only). Built on SPY + VIX + the
full stock panel. Shared helpers for the market and per-stock models.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, pandas as pd
import driver

SCR = os.path.dirname(__file__); OUT = "/home/user/academic_trading/results/sentiment_tuning"
SPLIT = pd.Timestamp("2016-01-01")   # pre-2016 tune (2008,2011,2015), post test (2018,2020,2022)


def market_features(corr_win=60, vmaw=50):
    spy = pd.read_parquet("price_data/SPY.parquet")["Close"]
    vix = pd.read_parquet("price_data/_VIX.parquet")["Close"]
    for s in (spy, vix): s.index = pd.to_datetime(s.index).normalize()
    # stock panel aligned to SPY dates for the correlation proxy
    panel = driver.build_panel_series(spy)
    dates = pd.DatetimeIndex(panel.dates); C = panel.C
    with np.errstate(invalid="ignore", divide="ignore"):
        Rs = np.vstack([np.full((1, C.shape[1]), np.nan), np.log(C[1:]/C[:-1])])   # stock log rets
    spy = spy.reindex(dates); vix = vix.reindex(dates)
    r = np.log(spy/spy.shift(1))
    # correlation proxy: index var / mean stock var over corr_win
    Rs_df = pd.DataFrame(Rs, index=dates)
    def corr_proxy(win):
        return (r.rolling(win).var() / Rs_df.rolling(win).var().mean(axis=1)).clip(0, 1)
    corr = corr_proxy(corr_win)
    df = pd.DataFrame(index=dates)
    df["px"] = spy; df["ret"] = r; df["vix"] = vix
    df["vspike"] = vix / vix.rolling(vmaw).mean()
    df["corr"] = corr
    # fast (20-day) correlation proxy: confirms systemic stress during fast crashes
    # (the 60-day proxy was one hundredth short at the 2025-04-08 low; see PHASE3_FINDINGS)
    df["corr20"] = corr_proxy(20)
    df["dd"] = (spy/spy.cummax() - 1)
    df["ext"] = spy/spy.rolling(200).mean() - 1
    df["dvix"] = vix.diff()
    df["rvc"] = r.rolling(corr_win).corr(vix.diff())   # return-vol corr (shown non-discriminating)
    return df


def fwd_return(px, h):
    return np.log(px.shift(-h)/px)        # forward h-day log return (look-ahead ONLY for scoring)


def zigzag(px, pct=0.15):
    """Ground-truth swing peaks/troughs: alternating extremes separated by >=pct
    reversals. Uses the full series (for SCORING only, never as a feature)."""
    p = px.to_numpy(float); n = len(p); idx = px.index
    if n == 0:
        return [], []
    piv = []
    trend = 0                      # 0 unknown, +1 up-leg, -1 down-leg
    hi = lo = p[0]; hi_i = lo_i = 0
    for i in range(1, n):
        if p[i] > hi: hi, hi_i = p[i], i
        if p[i] < lo: lo, lo_i = p[i], i
        if trend >= 0 and p[i] <= hi * (1 - pct):      # reversal down -> confirm the peak
            piv.append((hi_i, +1)); trend = -1; lo, lo_i = p[i], i
        elif trend <= 0 and p[i] >= lo * (1 + pct):    # reversal up -> confirm the trough
            piv.append((lo_i, -1)); trend = +1; hi, hi_i = p[i], i
    peaks = [idx[i] for i, d in piv if d == +1]
    troughs = [idx[i] for i, d in piv if d == -1]
    return peaks, troughs


if __name__ == "__main__":
    df = market_features()
    df.to_parquet(f"{SCR}/p3_market_features.parquet")
    print("features:", df.shape, df.index.min().date(), "..", df.index.max().date())
    print(df[["vix","vspike","corr","dd","ext","rvc"]].describe().round(3).to_string())
    pk, tr = zigzag(df["px"], 0.15)
    print(f"\nZigZag(15%) on SPY: {len(pk)} major peaks, {len(tr)} major troughs")
    print("recent peaks:", [d.date().isoformat() for d in pk[-4:]])
    print("recent troughs:", [d.date().isoformat() for d in tr[-4:]])
