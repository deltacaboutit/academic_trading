"""Simulated data so the tool can be tried without an internet connection.

The prices are a random walk with changing drift and volatility, so any
'signals' found in them are pure chance - which is itself a useful lesson
(Framework 9, step 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dates(n):
    end = pd.Timestamp.today().normalize()
    return pd.bdate_range(end=end, periods=n)


def demo_prices(n: int = 3800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regimes = rng.integers(60, 260, size=60)
    mu, sig = [], []
    for L in regimes:
        mu += [rng.normal(0.0004, 0.0012)] * int(L)
        sig += [rng.uniform(0.010, 0.028)] * int(L)
    mu = np.array(mu[:n]); sig = np.array(sig[:n])
    r = mu + sig * rng.standard_t(5, size=n) / np.sqrt(5 / 3)
    close = 40 * np.exp(np.cumsum(r))
    gap = rng.normal(0, 0.004, n)
    open_ = close * np.exp(-r + gap)            # roughly yesterday's close plus a gap
    open_[0] = close[0]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.lognormal(14, 0.35, n) * (1 + 25 * np.abs(r))
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol.round()},
                        index=_dates(n))


def demo_sentiment(dates: pd.DatetimeIndex, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    n = len(dates)
    px = demo_prices()["Close"].reindex(dates).to_numpy()
    ret = np.diff(np.log(px), prepend=np.log(px[0]))
    vix = np.empty(n); vix[0] = 18
    pc = np.empty(n); pc[0] = 0.65
    for t in range(1, n):
        vix[t] = max(9, vix[t - 1] + 0.05 * (18 - vix[t - 1]) - 120 * ret[t] + rng.normal(0, 0.8))
        pc[t] = max(0.2, pc[t - 1] + 0.1 * (0.65 - pc[t - 1]) - 2 * ret[t] + rng.normal(0, 0.05))
    iv = vix / 100 * 1.3 + rng.normal(0, 0.01, n)
    return {"vix": vix, "equity_pc": pc, "stock_pc": pc * rng.uniform(0.9, 1.1, n), "stock_iv": iv}
