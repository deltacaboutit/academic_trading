"""Clues ("features") the three indicators read, one row per trading day.

Each clue is a single number per day, built only from information available on
or before that day, so the feature table can be fed straight into a model or a
back-test without any risk of peeking at the future.

The clues are grouped into families, each with a plain reason it might help:

* **Stretch** - how far price has pulled away from its recent averages, in ATRs,
  plus RSI and the Bollinger position. Stretched prices tend to snap back.
* **Tiredness** - momentum fading while price still rises or falls (a
  "divergence"), and a flattening trend. A move often tires before it turns.
* **Volume** - heavy volume at an extreme (a climax), the balance of up-day
  versus down-day volume, and distance from the price where most shares have
  traded (the volume-profile "point of control").
* **Options** - distance to the call wall and put wall, which side of the gamma
  flip price sits on, extreme put/call ratios, and jumps in implied volatility.
  (Only a few months of history, so these are blank further back.)
* **Sector & market** - relative strength against the stock's sector fund and
  against the market, how tightly it tracks them, and the market's own trend.

Where a signed convention is natural, **positive means "top-like" (stretched up,
tiring at a high, near a ceiling) and negative means "bottom-like."** The
transparent score in forward_model.py relies on that convention; the statistical
model simply learns its own weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as I

Z_WIN = 252          # trailing window for standardising a clue (about one year)
Z_MIN = 60           # need at least this many days before a z-score is trusted


def _ztrail(x: np.ndarray, w: int = Z_WIN, minp: int = Z_MIN, clip: float = 5.0) -> np.ndarray:
    """Standardise a series using only its own past: subtract the trailing mean
    and divide by the trailing standard deviation. The result says how unusual
    today's value is for this stock, in standard deviations, and is comparable
    across stocks. Uses a window that includes today (causal), and is blank until
    ``minp`` days exist."""
    s = pd.Series(x, dtype=float)
    mu = s.rolling(w, min_periods=minp).mean()
    sd = s.rolling(w, min_periods=minp).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-clip, clip).to_numpy()


def _rolling_poc(high, low, vol, atr, window: int = 60, bins: int = 50) -> np.ndarray:
    """Distance from the volume-profile "point of control" (the price level where
    the most shares traded over the last ``window`` days), measured in ATRs and
    signed: positive when price is above that heavily-traded level.

    A true volume profile needs intraday prices, which Yahoo does not keep for
    long. This is the usual daily-bar approximation: each day's volume is spread
    evenly across that day's high-to-low range, and the busiest price band is the
    point of control. Price far above a heavy band can be over-extended; a heavy
    band below often acts as support (and above, as resistance)."""
    n = len(high)
    out = np.full(n, np.nan)
    for t in range(window, n):
        lo = low[t - window:t + 1]
        hi = high[t - window:t + 1]
        v = vol[t - window:t + 1]
        pmin, pmax = np.nanmin(lo), np.nanmax(hi)
        a = atr[t]
        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin or not np.isfinite(a) or a <= 0:
            continue
        edges = np.linspace(pmin, pmax, bins + 1)
        prof = np.zeros(bins)
        for j in range(len(lo)):
            if not (np.isfinite(lo[j]) and np.isfinite(hi[j]) and np.isfinite(v[j])) or hi[j] <= lo[j]:
                # all volume into the one bin containing the day's single price
                k = min(int((lo[j] - pmin) / (pmax - pmin) * bins), bins - 1) if pmax > pmin else 0
                prof[max(k, 0)] += v[j] if np.isfinite(v[j]) else 0.0
                continue
            lo_b = int((lo[j] - pmin) / (pmax - pmin) * bins)
            hi_b = int((hi[j] - pmin) / (pmax - pmin) * bins)
            lo_b = min(max(lo_b, 0), bins - 1)
            hi_b = min(max(hi_b, 0), bins - 1)
            span = hi_b - lo_b + 1
            prof[lo_b:hi_b + 1] += v[j] / span
        poc_bin = int(np.argmax(prof))
        out[t] = 0.5 * (edges[poc_bin] + edges[poc_bin + 1])    # the point-of-control price
    return out


def price_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """The stretch, tiredness and volume clues - everything that needs only the
    stock's own price and volume."""
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    n = len(close)
    atr = I.atr_filled(high, low, close, 14)
    atr_safe = np.where(atr > 0, atr, np.nan)

    sma20 = I.sma(close, 20)
    sma50 = I.sma(close, 50)
    rsi = I.rsi(close, 14)
    bb = I.bollinger(close, 20, 2.0)
    band = bb["up"] - bb["low"]
    pctb = np.where(band > 0, (close - bb["low"]) / band, 0.5)
    macd_line, macd_sig = I.macd(close)
    macd_hist = macd_line - macd_sig

    feats = {}
    # ---- stretch (positive = stretched up = top-like) ----
    feats["stretch_ma20"] = (close - sma20) / atr_safe
    feats["stretch_ma50"] = (close - sma50) / atr_safe
    feats["rsi"] = (rsi - 50.0) / 50.0                       # -1..+1, + = overbought
    feats["bb_pctb"] = (pctb - 0.5) * 2.0                    # ~ -1..+1, + = upper band
    feats["ret20_atr"] = (close - I.shift(close, 20)) / atr_safe

    # ---- tiredness ----
    L = 20
    price_chg = close - I.shift(close, L)
    rsi_chg = rsi - I.shift(rsi, L)
    # divergence: price up while RSI down -> + (top); price down while RSI up -> - (bottom)
    feats["mom_divergence"] = np.sign(price_chg) * (-rsi_chg) / 50.0
    feats["macd_hist"] = macd_hist / atr_safe
    # trend deceleration: second difference of sma20, signed with the trend
    slope = sma20 - I.shift(sma20, 5)
    accel = slope - I.shift(slope, 5)
    feats["trend_decel"] = -np.sign(slope) * accel / atr_safe   # + = decelerating in trend dir
    # days since the last 60-day high / low (0 = at a fresh extreme)
    hi60 = I.rolling_max(close, 60)
    lo60 = I.rolling_min(close, 60)
    feats["at_high60"] = (close >= hi60 * 0.999).astype(float)
    feats["at_low60"] = (close <= lo60 * 1.001).astype(float)

    # ---- volume ----
    rv = vol / I.sma(vol, 20)
    ret1 = np.sign(close - I.shift(close, 1))
    feats["rel_volume"] = rv - 1.0
    feats["climax"] = (rv - 1.0) * ret1                      # heavy vol up = + (buying climax/top)
    obv = I.obv(close, vol)
    feats["obv_slope"] = (obv - I.shift(obv, 20)) / (I.sma(vol, 20) * 20 + 1e-9)
    # volume-profile point of control -> signed distance in ATRs
    poc = _rolling_poc(high, low, vol, atr, window=60, bins=50)
    feats["dist_poc"] = (close - poc) / atr_safe

    out = pd.DataFrame(feats, index=df.index)
    return out


def options_features(df: pd.DataFrame, gex: dict | None, sentiment: dict | None) -> pd.DataFrame:
    """The options clues. ``gex`` is the dict from :func:`atrade.data.load_gex`
    (gamma flip and call/put walls per day); ``sentiment`` is the dict from
    :func:`atrade.data.load_sentiment` (put/call ratios and implied volatility).
    Columns are blank where the options history does not reach."""
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    n = len(close)
    atr = I.atr_filled(high, low, close, 14)
    atr_safe = np.where(atr > 0, atr, np.nan)
    out = pd.DataFrame(index=df.index)

    if gex is not None:
        flip = np.asarray(gex.get("gex_flip"), float) if gex.get("gex_flip") is not None else np.full(n, np.nan)
        cwall = np.asarray(gex.get("call_wall_price"), float) if gex.get("call_wall_price") is not None else np.full(n, np.nan)
        pwall = np.asarray(gex.get("put_wall_price"), float) if gex.get("put_wall_price") is not None else np.full(n, np.nan)
        # near the call wall from below = top-like (ceiling); near put wall from above = bottom-like (floor)
        out["to_call_wall"] = (cwall - close) / atr_safe        # small + = just under a ceiling
        out["to_put_wall"] = (close - pwall) / atr_safe         # small + = just above a floor
        # below the gamma flip, dealer hedging amplifies moves (unstable); above, it dampens them
        out["below_flip"] = np.where(np.isfinite(flip), np.sign(flip - close), np.nan)
        out["dist_flip"] = (close - flip) / atr_safe
    if sentiment:
        def arr_of(key):
            s = sentiment.get(key)
            if s is None:
                return None
            return s if isinstance(s, np.ndarray) else np.asarray(pd.Series(s).reindex(df.index), float)

        # put/call ratios: a high ratio means traders are buying protection (fear),
        # which is a contrarian, bottom-like signal -> negate so positive stays top-like.
        for src_key, name in (("stock_pc", "pc_volume_z"), ("stock_pc_oi", "pc_oi_z")):
            a = arr_of(src_key)
            if a is not None and np.isfinite(a).sum() > Z_MIN:
                out[name] = -_ztrail(a)
        iv = arr_of("stock_iv")
        if iv is not None and np.isfinite(iv).sum() > Z_MIN:
            out["iv_z"] = -_ztrail(iv)                           # high IV = fear = bottom-like
            out["iv_change"] = -_ztrail(iv - I.shift(iv, 5))     # IV spike = bottom-like
        # VIX is the market's fear gauge and has full history: high VIX is bottom-like.
        vix = arr_of("vix")
        if vix is not None and np.isfinite(vix).sum() > Z_MIN:
            out["vix_z"] = -_ztrail(vix)
    return out


def sector_features(df: pd.DataFrame, ticker: str, sector_etf_close: pd.Series | None,
                    market_close: pd.Series | None) -> pd.DataFrame:
    """The sector and market clues: relative strength against the stock's sector
    fund and the market, how closely it tracks the sector, and the market's own
    trend. Leading its sector while the sector leads the market is a healthier
    backdrop for a move to continue."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    r = close.pct_change()

    def align(bench):
        if bench is None:
            return None
        return bench.reindex(df.index).ffill(limit=5)

    etf = align(sector_etf_close)
    mkt = align(market_close)
    if etf is not None:
        rs = r.rolling(63).sum() - etf.pct_change().rolling(63).sum()
        out["rs_sector"] = rs.to_numpy()
        out["corr_sector"] = r.rolling(63).corr(etf.pct_change()).to_numpy()
        # is the sector itself strong? (sector fund vs its own 200d MA)
        out["sector_trend"] = (etf / etf.rolling(200).mean() - 1).to_numpy()
    if mkt is not None:
        rs = r.rolling(63).sum() - mkt.pct_change().rolling(63).sum()
        out["rs_market"] = rs.to_numpy()
        out["corr_market"] = r.rolling(63).corr(mkt.pct_change()).to_numpy()
        out["market_trend"] = (mkt / mkt.rolling(200).mean() - 1).to_numpy()
    return out


# Which columns are standardised vs already on a natural scale.
_ALREADY_SCALED = {"rsi", "bb_pctb", "at_high60", "at_low60", "below_flip", "market_trend",
                   "sector_trend", "corr_sector", "corr_market", "pc_volume_z", "pc_oi_z",
                   "iv_z", "iv_change", "vix_z"}


def build_features(df: pd.DataFrame, ticker: str = "", gex: dict | None = None,
                   sentiment: dict | None = None, sector_etf_close: pd.Series | None = None,
                   market_close: pd.Series | None = None, standardize: bool = True) -> pd.DataFrame:
    """Assemble every clue into one table, one row per day. Price and volume
    clues are always present; options and sector clues appear only when their
    data is supplied. With ``standardize`` the open-ended clues (distances,
    ratios) are turned into trailing z-scores so they are comparable across
    stocks; clues already on a natural scale (RSI, correlations, flags) are left
    as they are."""
    pv = price_volume_features(df)
    op = options_features(df, gex, sentiment)
    se = sector_features(df, ticker, sector_etf_close, market_close)
    feats = pd.concat([pv, op, se], axis=1)
    if standardize:
        for col in feats.columns:
            if col not in _ALREADY_SCALED:
                feats[col] = _ztrail(feats[col].to_numpy())
    return feats
