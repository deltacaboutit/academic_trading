"""Basic indicator formulas (book chapter 7). All functions take and return numpy arrays.

Every value on day t uses only prices up to and including day t.
"""
from __future__ import annotations

import numpy as np


def sma(x: np.ndarray, w: int) -> np.ndarray:
    """Simple moving average, SMA_{t|w} = (P_t + ... + P_{t-w+1}) / w."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    # any window containing a NaN is NaN
    nanmask = np.isnan(x).astype(float)
    cn = np.cumsum(np.insert(nanmask, 0, 0.0))
    bad = (cn[w:] - cn[:-w]) > 0
    out[w - 1:][bad] = np.nan
    return out


def lwma(x: np.ndarray, w: int) -> np.ndarray:
    """Linearly weighted moving average: newest day weight w, oldest weight 1."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    weights = np.arange(1, w + 1, dtype=float)
    denom = w * (w + 1) / 2
    for t in range(w - 1, len(x)):
        out[t] = np.dot(x[t - w + 1:t + 1], weights) / denom
    return out


def ema(x: np.ndarray, w: int, lam: float | None = None) -> np.ndarray:
    """Exponential moving average, started from a simple average (book eq. 7.4).

    EMA_t = (1 - lam) EMA_{t-1} + lam P_t, lam = 2/(w+1), first value = SMA of
    the first w valid values. Leading NaNs are skipped.
    """
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    lam = 2.0 / (w + 1) if lam is None else lam
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < w:
        return out
    i0 = valid[0]
    seed_end = i0 + w - 1
    if seed_end >= len(x):
        return out
    out[seed_end] = np.nanmean(x[i0:seed_end + 1])
    for t in range(seed_end + 1, len(x)):
        v = x[t]
        out[t] = out[t - 1] if np.isnan(v) else (1 - lam) * out[t - 1] + lam * v
    return out


def rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(w - 1, len(x)):
        out[t] = np.max(x[t - w + 1:t + 1])
    return out


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(w - 1, len(x)):
        out[t] = np.min(x[t - w + 1:t + 1])
    return out


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Value from k days earlier (NaN where not available)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k] if k > 0 else x
    return out


def macd(p: np.ndarray, fast=12, slow=26, sig=9):
    m = ema(p, fast) - ema(p, slow)
    s = ema(m, sig)
    return m, s


def momentum(p: np.ndarray, w=12, sig=9):
    mom = p - shift(p, w)
    return mom, sma(mom, sig)


def rsi(p: np.ndarray, w=14) -> np.ndarray:
    """RSI with plain w-day sums (book eq. 7.12). RSI = 100 if no down-moves."""
    p = np.asarray(p, float)
    d = np.diff(p, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    su = sma(up, w) * w
    sd = sma(dn, w) * w
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = su / sd
        out = 100 - 100 / (1 + rs)
    out[(sd == 0) & ~np.isnan(su)] = 100.0
    return out


def bollinger(p: np.ndarray, w=20, k=2.0):
    """Bollinger Bands, book version (eq. 7.16) plus the usual version.

    Book: sigma_t = sqrt( (1/w) * sum_{i=t-w+1..t} (P_i - mid_i)^2 ),
    where mid_i is each day's OWN w-day average.
    Usual: population std of the last w prices around today's average.
    """
    p = np.asarray(p, float)
    mid = sma(p, w)
    dev2 = (p - mid) ** 2
    sig_book = np.sqrt(sma(dev2, w))
    sig_usual = np.full(len(p), np.nan)
    if len(p) >= w:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        sig_usual[w - 1:] = swv(p, w).std(axis=1)
    return {
        "mid": mid,
        "up": mid + k * sig_book, "low": mid - k * sig_book,
        "up_usual": mid + k * sig_usual, "low_usual": mid - k * sig_usual,
    }


def roc(p: np.ndarray, w=12) -> np.ndarray:
    return 100 * (np.asarray(p, float) / shift(p, w) - 1)


def cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True on day t when a_t > b_t and a_{t-1} < b_{t-1} (book's cross rule)."""
    a1, b1 = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a > b) & (a1 < b1)


def cross_down(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a1, b1 = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a < b) & (a1 > b1)


# ---- Added from PatternFinder 2.5 (not in the book) -------------------------

def wilder_average(x: np.ndarray, w: int) -> np.ndarray:
    """Wilder's smoothing (Wilder 1978): the first value is the simple average of
    the first w values; after that each value is (previous x (w - 1) + new) / w."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < w:
        return out
    i0 = valid[0]
    out[i0 + w - 1] = np.mean(x[i0:i0 + w])
    for t in range(i0 + w, len(x)):
        out[t] = out[t - 1] if np.isnan(x[t]) else (out[t - 1] * (w - 1) + x[t]) / w
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder's true range: the largest of today's high - low, |high - yesterday's
    close| and |low - yesterday's close|. The first day has no previous close."""
    h, l, c = (np.asarray(a, float) for a in (high, low, close))
    pc = shift(c)
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(high, low, close, w: int = 14) -> np.ndarray:
    """Average True Range (Wilder 1978): the typical size of one day's price range.
    Empty for the first w - 1 days."""
    return wilder_average(true_range(high, low, close), w)


def atr_filled(high, low, close, w: int = 14) -> np.ndarray:
    """ATR with the first w - 1 days filled by the running average of the true range."""
    tr = true_range(high, low, close)
    out = atr(high, low, close, w)
    run = np.cumsum(np.nan_to_num(tr)) / np.arange(1, len(tr) + 1)
    return np.where(np.isnan(out), run, out)


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume (Granville 1963): a running total of volume. A day that
    closes higher adds its volume; a day that closes lower subtracts it."""
    c = np.asarray(close, float)
    step = np.sign(np.nan_to_num(np.diff(c, prepend=c[0]))) * np.nan_to_num(np.asarray(volume, float))
    return np.cumsum(step)
