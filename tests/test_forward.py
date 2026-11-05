"""Tests for the forward-looking indicators (tops, bottoms, continuation),
their features, models and the sector/correlation layer.

The most important checks are that the labels never peek at the future and that
the features are causal (a value computed today does not change when tomorrow's
data arrives) - without those, any back-test result would be meaningless.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ATRADE_OFFLINE", "1")

from atrade import forward_features as FF
from atrade import forward_labels as FL
from atrade import forward_model as M
from atrade import sectors as S
from atrade.config import DEFAULTS


def _ohlc_from_close(close, vol=None):
    close = np.asarray(close, float)
    idx = pd.bdate_range("2000-01-03", periods=len(close))
    high = close * 1.01
    low = close * 0.99
    if vol is None:
        vol = np.full(len(close), 1e6)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": vol}, index=idx)


# ----------------------------------------------------------------- labels ---

def test_labels_have_no_lookahead():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
    df = _ohlc_from_close(close)
    state = np.sign(df["Close"].diff(50).fillna(0).to_numpy())
    it, ib, last = FL.label_tops_bottoms(df, window=10, horizon=40)
    cont, lastc = FL.label_continuation(df, state, horizon=20)
    # the final max(window,horizon) days cannot be labelled yet
    assert np.isnan(it[last + 1:]).all()
    assert np.isnan(ib[last + 1:]).all()
    assert np.isnan(cont[lastc + 1:]).all()


def test_top_label_fires_on_a_clear_peak():
    # rise to a sharp peak, then a decisive fall -> the peak day is a top
    up = np.linspace(100, 140, 60)
    down = np.linspace(140, 100, 60)[1:]     # drop the duplicate 140 so the peak is a single point
    close = np.r_[np.full(30, 100.0), up, down, np.full(30, 100.0)]
    df = _ohlc_from_close(close)
    it, ib, last = FL.label_tops_bottoms(df, window=10, drop_atr=2.0, horizon=40)
    peak = int(np.argmax(close))
    # some day at/near the peak is flagged a top, and none near it is flagged a bottom
    assert np.nansum(it[peak - 5:peak + 5]) >= 1
    assert np.nansum(ib[peak - 5:peak + 5]) == 0


def test_labels_are_sparse():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 3000)))
    df = _ohlc_from_close(close)
    it, ib, _ = FL.label_tops_bottoms(df)
    top_rate = np.nanmean(it)
    assert 0.0 < top_rate < 0.15          # tops are rare, not every other day


# --------------------------------------------------------------- features ---

def test_features_are_causal():
    # a feature computed on a prefix must equal the prefix of the full computation
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
    df = _ohlc_from_close(close)
    full = FF.price_volume_features(df)
    cut = 600
    pre = FF.price_volume_features(df.iloc[:cut])
    for col in ["stretch_ma20", "rsi", "macd_hist", "rel_volume"]:
        a = full[col].to_numpy()[:cut]
        b = pre[col].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both], atol=1e-8), col


def test_build_features_columns_and_nan_tolerance():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    df = _ohlc_from_close(close)
    feats = FF.build_features(df)                 # no options / sector supplied
    assert len(feats) == len(df)
    for c in ["stretch_ma20", "rsi", "climax", "dist_poc"]:
        assert c in feats.columns
    # options columns are simply absent when no options data is given
    assert "to_call_wall" not in feats.columns


# ----------------------------------------------------------------- models ---

def test_logit_recovers_known_weights():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, 3))
    true = np.array([-0.5, 1.5, -1.0, 0.8])
    p = 1 / (1 + np.exp(-(true[0] + X @ true[1:])))
    y = (rng.random(3000) < p).astype(float)
    w = M.fit_logit(X, y, l2=0.1)
    assert np.allclose(w, true, atol=0.2)
    pr = M.predict_proba(w, X)
    assert pr.min() >= 0 and pr.max() <= 1


def test_impute_fills_missing_with_zero():
    X = np.array([[1.0, np.nan], [np.nan, 2.0]])
    out = M._impute(X)
    assert out[0, 1] == 0.0 and out[1, 0] == 0.0


def test_continuation_features_flip_with_trend():
    feats = pd.DataFrame({"stretch_ma20": [2.0, 2.0], "rsi": [0.5, 0.5],
                          "bb_pctb": [1.0, 1.0], "ret20_atr": [1.0, 1.0],
                          "mom_divergence": [0.0, 0.0], "trend_decel": [0.0, 0.0],
                          "climax": [0.0, 0.0], "dist_poc": [0.0, 0.0],
                          "stretch_ma50": [1.0, 1.0], "rel_volume": [0.0, 0.0]})
    up = M.continuation_features(feats, np.array([1, 1]))
    dn = M.continuation_features(feats, np.array([-1, -1]))
    # counter-trend stretch has opposite sign in an up- vs down-trend
    assert up["counter_stretch_ma20"].iloc[0] == -dn["counter_stretch_ma20"].iloc[0]
    assert up["counter_stretch_ma20"].iloc[0] > 0


def test_expected_range_sane():
    close = 100 + np.cumsum(np.random.default_rng(5).normal(0, 1, 400))
    df = _ohlc_from_close(close)
    r = M.expected_range(df, iv_annual=0.30, horizon=20)
    assert r["low"] < df["Close"].iloc[-1] < r["high"]
    assert r["sigma_pct"] > 0
    # implied 30% vol over 20 trading days ~ 8-9% one-sigma
    assert 0.05 < r["sigma_pct"] < 0.12


# ---------------------------------------------------------------- sectors ---

@pytest.mark.skipif(S._close("XLF") is None or S._close("JPM") is None,
                    reason="sector ETF / stock price files not present")
def test_assign_sector_matches_obvious_case():
    asg = S.assign_sector("JPM")
    assert asg["etf"] == "XLF"           # a bank tracks the financials fund


@pytest.mark.skipif(S._close("XLK") is None, reason="sector ETF files not present")
def test_correlation_report_keys():
    rep = S.correlation_report("JPM")
    assert "sector" in rep and "market_breadth" in rep
    assert set(rep["market_breadth"]) >= {"above_ma50", "above_ma200"}


# --------------------------------------------------------------- end to end -

@pytest.mark.skipif(S._close("SPY") is None, reason="price files not present")
def test_forward_estimate_smoke():
    from atrade import forward as FW
    tkr = "JPM"
    if __import__("atrade.data", fromlist=["repo_prices"]).repo_prices(tkr) is None:
        pytest.skip("no offline prices for test ticker")
    est = FW.forward_estimate(tkr)
    for key in ["ticker", "top", "bottom", "continuation", "expected_range",
                "key_levels", "pressure", "correlation"]:
        assert key in est
    txt = FW.render_text(est)
    assert tkr in txt
