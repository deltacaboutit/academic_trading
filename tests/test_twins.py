"""Checks of the random-twin test (twins.py) and its statistics (edge_stats.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade.config import DEFAULTS
from atrade.edge_stats import benjamini_hochberg, check_report, clustered_se, holm, two_sided_p, variant_report
from atrade.twins import pattern_stop, simulate, twin_trades

S = dict(DEFAULTS)


def _bars(c, o=None, h=None, l=None):
    c = np.asarray(c, float)
    o = c.copy() if o is None else np.asarray(o, float)
    return o, (np.maximum(o, c) + 0.1 if h is None else np.asarray(h, float)), (np.minimum(o, c) - 0.1 if l is None else np.asarray(l, float)), c


def test_target_is_hit_on_the_days_high_and_costs_are_taken():
    o, h, l, c = _bars([100, 100, 101, 102, 103])
    h[3] = 105.0                                           # the high reaches the target; the close does not
    ret, j, why = simulate(o, h, l, c, 0, +1, 105.0, 95.0, 10, cost=0.001)
    assert (j, why) == (3, "target") and ret == pytest.approx(0.05 - 0.002)


def test_gap_through_the_stop_exits_at_the_open():
    o, h, l, c = _bars([100, 100, 99, 90, 90], o=[100, 100, 100, 91, 90])
    ret, j, why = simulate(o, h, l, c, 0, +1, 110.0, 95.0, 10, cost=0.0)
    assert (j, why) == (3, "stop") and ret == pytest.approx(91 / 100 - 1)   # filled at the open, below the stop


def test_stop_counts_first_when_both_are_hit_on_one_day():
    o, h, l, c = _bars([100, 100, 100, 100])
    h[2], l[2] = 111.0, 94.0
    assert simulate(o, h, l, c, 0, +1, 110.0, 95.0, 10, cost=0.0)[2] == "stop"


def test_trade_is_skipped_if_the_open_is_already_past_a_level():
    o, h, l, c = _bars([100, 100, 100], o=[100, 111, 100])
    assert simulate(o, h, l, c, 0, +1, 110.0, 95.0, 10) is None


def test_line_stop_after_two_closes_back_across_the_line():
    o, h, l, c = _bars([100, 100, 99, 98.5, 98, 97])
    line = np.full(10, 0.99)                               # the broken line sits 1% below the entry
    ret, j, why = simulate(o, h, l, c, 0, +1, 120.0, 80.0, 10, line_rel=line, cost=0.0)
    assert (j, why) == (4, "line")                         # day 2 closes ON the line; days 3 and 4 close below it


def test_holm_and_benjamini_hochberg():
    p = [0.01, 0.04, 0.03, 0.20]
    assert list(holm(p)) == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert list(benjamini_hochberg(p)) == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20])
    assert two_sided_p(1.96) == pytest.approx(0.05, abs=1e-3) and two_sided_p(2.0, 10) > two_sided_p(2.0)


def test_clustered_se_equals_the_plain_one_when_every_trade_is_its_own_month():
    x = np.random.default_rng(1).normal(size=50)
    plain = x.std(ddof=1) / np.sqrt(50)
    assert clustered_se(x, np.arange(50)) == pytest.approx(plain, rel=1e-9)
    assert clustered_se(x, np.repeat(np.arange(5), 10)) != pytest.approx(plain)


def test_a_check_only_helps_with_a_real_effect_in_both_halves():
    rng = np.random.default_rng(2)
    n = 400
    dates = pd.date_range("2005-01-01", periods=n, freq="7D").strftime("%Y-%m-%d")
    flag = rng.integers(0, 2, n)
    edge = rng.normal(0, 1, n) + 1.0 * flag                  # passing the check adds 1% per trade
    t = pd.DataFrame({"signal_date": dates, "edge_pct": edge, "chk_volume_dryup": flag,
                      "chk_rsi_ok": rng.integers(0, 2, n)})
    rep = check_report(t).set_index("check")
    assert rep.loc["volume_dryup", "verdict"] == "HELPS"
    assert rep.loc["rsi_ok", "verdict"] == "no clear effect"
    t["chk_rsi_ok"] = np.r_[np.ones(n - 5), np.zeros(5)]    # only 5 trades failed it
    assert check_report(t).set_index("check").loc["rsi_ok", "verdict"] == "too few"


def test_twin_trades_on_demo_prices():
    from atrade.cleaning import clean
    from atrade.demo import demo_prices
    from atrade.patterns import all_patterns
    from atrade.trend import trend_state
    df, _, _ = clean(demo_prices(), S)
    c = df["Close"].to_numpy(float)
    sigs = all_patterns(c, S, trend_state(c, S), df["High"].to_numpy(float), df["Low"].to_numpy(float))
    a = twin_trades("DEMO", df, sigs, S)
    b = twin_trades("DEMO", df, sigs, S)
    assert len(a) >= 10 and a["edge_pct"].equals(b["edge_pct"])           # the same random days every run
    assert np.allclose(a["edge_pct"], a["return_pct"] - a["random_return_pct"])
    assert (a["twins"] > 0).all()
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    from atrade import indicators as ind
    atr = ind.atr_filled(h, l, c)
    for g in sigs:
        ps = pattern_stop(g, h, l, atr)
        if ps is not None and g.rule != "Rectangle":
            assert (ps < c[g.idx]) if g.direction > 0 else (ps > c[g.idx]), g.rule   # the stop is on the losing side
    v = variant_report(a)
    assert set(v["variant"]) == {"kr_exit", "gap_stop", "pattern_stop"}
