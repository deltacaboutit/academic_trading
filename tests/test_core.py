"""Checks of the formulas against the framework and the book, and a test that no
signal uses information from the future. Run with:  python -m pytest -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade import indicators as ind
from atrade.config import DEFAULTS
from atrade.demo import demo_prices, demo_sentiment
from atrade.patterns import all_patterns, double_tops_bottoms, head_and_shoulders, rounding
from atrade.turning_points import pips, rolling_window
from atrade.validation import bernoulli_test
from atrade.zones import build_bins

S = dict(DEFAULTS)


def test_ema_starts_from_sma():
    x = np.arange(1, 31, dtype=float)
    e = ind.ema(x, 10)
    assert np.isnan(e[8]) and e[9] == pytest.approx(x[:10].mean())
    lam = 2 / 11
    assert e[10] == pytest.approx((1 - lam) * e[9] + lam * x[10])


def test_lwma():
    x = np.array([1, 2, 3, 4], float)
    assert ind.lwma(x, 3)[-1] == pytest.approx((3 * 4 + 2 * 3 + 1 * 2) / 6)


def test_rsi_no_down_moves_is_100_and_uses_plain_sums():
    x = np.arange(1, 40, dtype=float)
    assert ind.rsi(x, 14)[-1] == 100
    y = np.array([10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16], float)
    up = sum(max(d, 0) for d in np.diff(y)); dn = sum(-min(d, 0) for d in np.diff(y))
    assert ind.rsi(y, 14)[-1] == pytest.approx(100 - 100 / (1 + up / dn))


def test_bollinger_book_sigma_uses_each_days_own_mean():
    rng = np.random.default_rng(0)
    p = 100 + np.cumsum(rng.normal(0, 1, 80))
    bb = ind.bollinger(p, 20, 2)
    t = 70
    mids = [p[i - 19:i + 1].mean() for i in range(t - 19, t + 1)]
    sig = math.sqrt(np.mean([(p[i] - m) ** 2 for i, m in zip(range(t - 19, t + 1), mids)]))
    assert bb["up"][t] == pytest.approx(mids[-1] + 2 * sig)
    assert bb["up_usual"][t] == pytest.approx(mids[-1] + 2 * p[t - 19:t + 1].std())


def test_rolling_window_strict_ties():
    p = np.array([1, 2, 3, 5, 5, 3, 2, 1, 0, 1, 2], float)
    tp = rolling_window(p, 2)
    assert not ((tp["kind"] == "P") & tp["idx"].isin([3, 4])).any()   # flat top: no peak
    p2 = np.array([1, 2, 3, 6, 5, 3, 2, 1, 0, 1, 2], float)
    tp2 = rolling_window(p2, 2)
    row = tp2[tp2["kind"] == "P"].iloc[0]
    assert row["idx"] == 3 and row["known"] == 5


def test_pips_picks_obvious_points():
    y = np.array([0, 1, 2, 3, 10, 3, 2, 1, 0], float)
    assert pips(y, 3) == [0, 4, 8]


def test_bins_centre_end_points_and_rounding():
    prices = np.array([10, 10.2, 13, 13.1, 20], float)
    bounds, counts, xh = build_bins(prices, 0.03)
    n = math.log(bounds[-1] / bounds[0]) / math.log(1.03)
    assert len(bounds) - 1 == round(n)
    assert counts.sum() == len(prices)
    # lowest point sits near the centre of the first bin
    assert (bounds[0] + bounds[1]) / 2 == pytest.approx(10, rel=0.002)


def test_bernoulli_examples_from_design():
    assert bernoulli_test(484, 501)["z"] == pytest.approx(20.9, abs=0.05)
    assert (4066 - 4062.54) / 39.81 == pytest.approx(0.087, abs=0.002)   # Appendix note on Table 4.4


def _path(knots, start=0):
    """Piecewise-linear prices through (day, price) knots."""
    days = [k[0] for k in knots]; px = [k[1] for k in knots]
    return np.interp(np.arange(days[-1] + 1), days, px)


def test_head_and_shoulders_detected_and_breaks_neckline():
    p = _path([(0, 80), (20, 100), (40, 90), (60, 110), (80, 90), (100, 100), (120, 80)])
    sigs = head_and_shoulders(p, S, np.zeros(len(p)))
    hs = [x for x in sigs if x.rule == "Head-and-shoulders top"]
    assert len(hs) == 1
    g = hs[0]
    assert g.direction == -1 and p[g.idx] < 90 and g.idx >= 100 + S["w_hs"]
    assert g.target == pytest.approx(90 - 20)           # neckline 90, head height 20


def test_double_bottom_detected():
    p = _path([(0, 120), (30, 100), (60, 115), (90, 101), (130, 130)])
    sigs = double_tops_bottoms(p, S, np.zeros(len(p)))
    db = [x for x in sigs if x.rule == "Double bottom"]
    assert db and db[0].direction == +1 and p[db[0].idx] > 115


def test_rounding_bottom_detected():
    t = np.arange(0, 61)
    bowl = 100 - 20 * np.sin(np.pi * t / 60)                  # smooth saucer
    p = np.concatenate([np.linspace(90, 100, 15)[:-1] + np.r_[np.zeros(13), 0.5], bowl, np.linspace(100, 110, 20)])
    p[14] = 100.5                                              # left rim is a clear local peak
    sigs = rounding(p, S, np.zeros(len(p)))
    assert any(x.rule == "Rounding bottom" for x in sigs)


@pytest.mark.parametrize("cut", [1500, 2400, 3300])
def test_no_look_ahead(cut):
    """Signals dated on or before day `cut` must be identical whether or not
    the data after `cut` exists."""
    from atrade.cleaning import clean
    from atrade.gauges import gauge_signals
    from atrade.sentiment import sentiment_signals
    from atrade.trend import trend_signals, trend_state
    from atrade.zones import zone_analysis

    raw = demo_prices()
    df, _, _ = clean(raw, S)
    sent = demo_sentiment(df.index)

    def run(d):
        c = d["Close"].to_numpy(float)
        st = trend_state(c, S)
        out = trend_signals(d, S) + gauge_signals(c, S) + zone_analysis(c, S)[0] + all_patterns(c, S, st)
        out += sentiment_signals({k: v[:len(d)] for k, v in sent.items()}, d, S)[0]
        return st, sorted((x.rule, x.idx, x.direction) for x in out if x.idx <= cut)

    st_full, full = run(df)
    st_cut, part = run(df.iloc[:cut + 1])
    assert full == part
    assert np.array_equal(st_full[:cut + 1], st_cut)


def test_bull_flag_detected():
    from atrade.patterns import flags_pennants_wedges
    p = _path([(0, 50), (40, 100), (43, 96), (46, 98), (49, 94), (60, 110)])
    sigs = [x for x in flags_pennants_wedges(p, S, np.zeros(len(p))) if x.rule == "Bull flag"]
    assert sigs, "bull flag not found"
    g = sigs[0]
    assert g.idx >= 49 + S["w_flag"] and g.direction == 1
    assert g.target == pytest.approx(g.line_value(g.idx) + (100 - 50))
