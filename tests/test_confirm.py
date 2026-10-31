"""Checks of ATR, OBV, key reversal days and the 21 textbook checks (confirm.py),
including a test that no check looks at prices after the day it is judged on."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade import indicators as ind
from atrade.config import DEFAULTS
from atrade.confirm import CHECK_KEYS, CHECKS, NEEDS_BREAKOUT, PROVEN, Context, key_reversal_days, run_checks
from atrade.demo import demo_prices

S = dict(DEFAULTS)


def test_atr_uses_wilders_smoothing():
    h = np.array([11, 12, 13, 12, 14], float); l = np.array([9, 10, 11, 10, 12], float)
    c = np.array([10, 11, 12, 11, 13], float)
    tr = ind.true_range(h, l, c)
    assert list(tr) == [2, 2, 2, 2, 3]                     # e.g. day 4: max(14-12, |14-11|, |12-11|) = 3
    a = ind.atr(h, l, c, 3)
    assert np.isnan(a[1]) and a[2] == pytest.approx(2.0)
    assert a[4] == pytest.approx((2.0 * 2 + 3) / 3)        # (previous x 2 + new) / 3


def test_obv_adds_up_days_and_subtracts_down_days():
    c = np.array([10, 11, 11, 10, 12], float); v = np.array([5, 1, 2, 3, 4], float)
    assert list(ind.obv(c, v)) == [0, 1, 1, -2, 2]


def test_key_reversal_day():
    n = 15
    c = np.linspace(110, 100, n); o = c + 0.2; h = c + 0.5; l = c - 0.5; v = np.full(n, 100.0)
    o[-1], l[-1], c[-1], h[-1], v[-1] = 99.5, 98.0, 101.5, 102.0, 300.0   # new low, opens lower, closes higher
    bull, bear, sbull, sbear = key_reversal_days(o, h, l, c, v)
    assert bull[-1] and sbull[-1] and not bear.any()


def test_every_check_is_defined_once():
    assert len(CHECKS) == 21 and len(set(CHECK_KEYS)) == 21
    assert NEEDS_BREAKOUT <= set(CHECK_KEYS) and PROVEN <= set(CHECK_KEYS)


@pytest.fixture(scope="module")
def demo():
    from atrade.cleaning import clean
    from atrade.patterns import all_patterns
    from atrade.trend import trend_state
    df, _, _ = clean(demo_prices(), S)
    market = pd.Series(np.exp(np.cumsum(np.random.default_rng(5).normal(0.0003, 0.01, len(df)))) * 100, index=df.index)
    c = df["Close"].to_numpy(float)
    setups = []
    sigs = all_patterns(c, S, trend_state(c, S), df["High"].to_numpy(float), df["Low"].to_numpy(float), setups=setups)
    return df, market, sigs, setups


def _same(a, b):
    """Two sets of check results are equal (a blank number counts as equal to a blank number)."""
    norm = lambda r: {k: (x["applies"], x["passed"], None if math.isnan(x["value"]) else round(x["value"], 9), x["text"])
                      for k, x in r.items()}
    return norm(a) == norm(b)


def test_checks_never_look_ahead(demo):
    """Judging a check on day i gives the same answer with or without the prices after day i."""
    df, market, sigs, setups = demo
    full = Context(df, market)
    picked = [g for g in sigs if 300 < g.idx < len(df) - 5][::3][:12]
    assert len(picked) >= 6
    for g in picked:
        part = Context(df.iloc[:g.idx + 1], market)
        assert _same(run_checks(full, g), run_checks(part, g)), g.rule
    for u in [u for u in setups if u.resolved is not None and u.resolved > u.known + 2][:8]:
        i = u.known + 1                      # judged while it was still forming
        part = Context(df.iloc[:i + 1], market)
        assert _same(run_checks(full, u, i), run_checks(part, u, i)), u.rule


def test_forming_patterns_leave_breakout_checks_blank(demo):
    df, market, sigs, setups = demo
    ctx = Context(df, market)
    u = next(u for u in setups if u.direction != 0)
    chk = run_checks(ctx, u, u.known)
    assert all(chk[k]["passed"] is None for k in NEEDS_BREAKOUT if chk[k]["applies"])
    g = next(g for g in sigs if g.idx > 400)
    chk = run_checks(ctx, g)
    assert chk["market_trend"]["passed"] is not None and chk["rsi_ok"]["passed"] is not None
    nomkt = run_checks(Context(df, None), g)
    assert nomkt["market_trend"]["passed"] is None and nomkt["relative_strength"]["passed"] is None
