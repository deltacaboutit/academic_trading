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

    def as_of_cut(u):
        # a setup's status as it looked on day `cut`: anything decided later was still forming then
        done = u.resolved is not None and u.resolved <= cut
        return (u.rule, u.known, tuple((a, round(b, 9)) for a, b, _ in u.points),
                u.status if done else "Forming", u.resolved if done else None, u.broke_dir if done else 0)

    def run(d):
        c = d["Close"].to_numpy(float)
        st = trend_state(c, S)
        setups = []
        out = trend_signals(d, S) + gauge_signals(c, S) + zone_analysis(c, S)[0]
        out += all_patterns(c, S, st, d["High"].to_numpy(float), d["Low"].to_numpy(float), setups=setups)
        out += sentiment_signals({k: v[:len(d)] for k, v in sent.items()}, d, S)[0]
        return (st, sorted((x.rule, x.idx, x.direction) for x in out if x.idx <= cut),
                sorted(as_of_cut(u) for u in setups if u.known <= cut))

    st_full, full, set_full = run(df)
    st_cut, part, set_cut = run(df.iloc[:cut + 1])
    assert full == part
    assert np.array_equal(st_full[:cut + 1], st_cut)
    # forming patterns: what was known on day `cut` is the same with or without the later data
    assert set_full == set_cut and set_cut


def test_bull_flag_detected():
    from atrade.patterns import flags_pennants_wedges
    p = _path([(0, 50), (40, 100), (43, 96), (46, 98), (49, 94), (60, 110)])
    sigs = [x for x in flags_pennants_wedges(p, S, np.zeros(len(p))) if x.rule == "Bull flag"]
    assert sigs, "bull flag not found"
    g = sigs[0]
    assert g.idx >= 49 + S["w_flag"] and g.direction == 1
    assert g.target == pytest.approx(g.line_value(g.idx) + (100 - 50))


@pytest.mark.parametrize("knots,mirror,rule,d,target", [
    ([(0, 70), (10, 100), (25, 80), (40, 100.5), (55, 85), (70, 99.8), (80, 90), (100, 115)], False,
     "Ascending triangle", +1, 100.5 + (100.5 - 77.5)),            # flat top 100.5; rising line is 77.5 on day 10
    ([(0, 70), (10, 100), (25, 80), (40, 100.5), (55, 85), (70, 99.8), (80, 90), (100, 115)], True,
     "Descending triangle", -1, 200 - (100.5 + (100.5 - 77.5))),   # the same shape upside down
    ([(0, 80), (10, 110), (25, 100), (40, 110.5), (55, 100.5), (70, 109.8), (78, 103), (95, 120)], False,
     "Rectangle", +1, 110.5 + (110.5 - 100)),
])
def test_triangles_and_rectangle_detected(knots, mirror, rule, d, target):
    from atrade.patterns import triangles_rectangles
    p = _path(knots)
    if mirror:
        p = 200 - p
    sigs = [x for x in triangles_rectangles(p, S, np.zeros(len(p))) if x.rule == rule]
    assert len(sigs) == 1, f"{rule} not found exactly once"
    g = sigs[0]
    assert g.direction == d and g.idx >= 70 + S["w_rect"]          # only after the last turning point is confirmed
    assert (p[g.idx] - g.line_value(g.idx)) * d > 0                 # the close is beyond the broken line
    assert g.target == pytest.approx(target)


def test_symmetrical_triangle_needs_converging_lines():
    from atrade.patterns import triangles_rectangles
    p = _path([(0, 70), (10, 130), (25, 80), (40, 123), (55, 87), (70, 116), (78, 100), (95, 125)])
    sigs = triangles_rectangles(p, S, np.zeros(len(p)))
    assert [x.rule for x in sigs] == ["Symmetrical triangle"] and sigs[0].direction == +1
    q = _path([(0, 70), (10, 130), (25, 80), (40, 130.5), (55, 80.5), (70, 129.8), (78, 100), (95, 140)])
    assert not [x for x in triangles_rectangles(q, S, np.zeros(len(q))) if "triangle" in x.rule]


def test_cup_and_handle_detected():
    from atrade.patterns import cup_and_handle
    t = np.arange(0, 81)
    cup = 100 - 25 * np.sin(np.pi * t / 80)                       # round cup, 25% deep
    p = np.concatenate([np.linspace(85, 99, 15), [100.5], cup[1:-1], [100],
                        np.linspace(99, 94, 8)[1:], np.linspace(94, 105, 10)[1:]])
    sigs = cup_and_handle(p, S, np.zeros(len(p)))
    assert len(sigs) == 1
    g = sigs[0]
    assert g.direction == +1 and p[g.idx] > 100 and p[g.idx - 1] <= 100
    assert g.target == pytest.approx(100 + (100 - 75))             # right rim plus the cup's depth
    v = np.concatenate([np.linspace(85, 99, 15), [100.5], np.linspace(100, 75, 41)[1:], np.linspace(75, 100, 40)[1:],
                        np.linspace(99, 94, 8)[1:], np.linspace(94, 105, 10)[1:]])
    assert not cup_and_handle(v, S, np.zeros(len(v)))              # a V-shaped cup is not round enough


def test_island_top_and_bottom_detected():
    from atrade.patterns import all_patterns, islands
    c = np.r_[np.linspace(80, 100, 30), [104, 105, 104.5], np.linspace(98, 90, 10)]
    h, lo = c + 0.5, c - 0.5
    top = islands(h, lo, c, S, np.zeros(len(c)))
    assert [(x.rule, x.direction, x.idx) for x in top] == [("Island top", -1, 33)]
    assert top[0].target == pytest.approx(103.5 - (105.5 - 103.5))  # island edge minus the island's height
    bot = islands(200 - lo, 200 - h, 200 - c, S, np.zeros(len(c)))
    assert [(x.rule, x.direction, x.idx) for x in bot] == [("Island bottom", +1, 33)]
    assert bot[0].target == pytest.approx(200 - top[0].target)
    h2, l2 = h.copy(), lo.copy(); h2[32], l2[32] = 104.0, 100.0     # the island falls back into the first gap
    assert not islands(h2, l2, c, S, np.zeros(len(c)))
    assert not [x for x in all_patterns(c, S, np.zeros(len(c))) if x.rule.startswith("Island")]  # closes only


def test_download_prices_layout_and_retry_logic(tmp_path):
    """The price downloader with a fake Yahoo: adjusted prices, raw close, manifest, retries."""
    from atrade import download_prices as DP
    idx = pd.bdate_range("2025-01-01", periods=30)
    raw = pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 101.0, "Adj Close": 50.5,
                        "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    calls = {"BAD": 0}

    def fake(t):
        if t == "BAD":
            calls["BAD"] += 1
            raise RuntimeError("no data")
        return DP.to_table(raw)
    rc = DP.main(["--tickers", "AAPL", "BRK.B", "^VIX", "BAD"], fetcher=fake, out=tmp_path)
    assert rc == 1
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["AAPL.parquet", "BRK.B.parquet", "_VIX.parquet"]
    df = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert df["Close"].iloc[0] == 50.5 and df["Open"].iloc[0] == 50.0 and df["Close_raw"].iloc[0] == 101.0
    man = pd.read_csv(tmp_path / "manifest.csv").set_index("ticker")
    assert man.loc["AAPL", "days"] == 30 and man.loc["AAPL", "bad_bars"] == 0
    assert "no data" in man.loc["BAD", "error"]
    # --missing-only retries only the failed ticker
    DP.main(["--tickers", "AAPL", "BAD", "--missing-only"], fetcher=fake, out=tmp_path)
    assert calls["BAD"] == 2


# ---------------------------------------------------------- forming patterns --

def _setups_and_signals(c, high=None, low=None):
    setups = []
    sigs = all_patterns(c, S, np.zeros(len(c)), high, low, setups=setups)
    return setups, sigs


def test_setups_match_signals_and_never_change_them():
    """A setup that "broke out" is exactly a pattern Signal, and asking for
    setups never changes the signals (so the rating and back-test are untouched)."""
    from atrade.cleaning import clean
    df, _, _ = clean(demo_prices(), S)
    walks = [100 * np.exp(np.cumsum(np.random.default_rng(k).normal(0, 0.015, 2500))) for k in range(3)]
    series = [df["Close"].to_numpy(float)] + walks
    for c in series:
        setups, sigs = _setups_and_signals(c)
        plain = all_patterns(c, S, np.zeros(len(c)))
        assert [(x.rule, x.idx, x.direction, x.target) for x in sigs] == \
               [(x.rule, x.idx, x.direction, x.target) for x in plain]
        assert setups and all(isinstance(u.status, str) for u in setups)
        sk = {(x.rule, x.idx, x.direction) for x in sigs if not x.rule.startswith(("Rounding", "Island"))}
        bk = {(u.rule, u.resolved, u.broke_dir) for u in setups if u.status == "Broke out"}
        assert bk <= sk
        # a signal without a "broke out" setup is one whose setup failed by passing its extreme first
        for rule, idx, d in sk - bk:
            assert any(u.rule == rule and u.status == "Failed" and "extreme" in u.reason and u.resolved < idx
                       for u in setups)
        for u in setups:
            assert u.known <= (u.resolved if u.resolved is not None else len(c) - 1)
            assert u.status != "Forming" or u.deadline >= len(c) - 1


def test_double_top_forming_failed_expired_and_broke_out():
    from atrade.forming import forming_now
    shape = [(0, 80), (30, 100), (45, 85), (60, 100.5), (80, 90)]   # two peaks, trough 85, no break yet
    setups, _ = _setups_and_signals(_path(shape))
    now = [u for u in forming_now(setups) if u.rule == "Double top"]
    assert len(now) == 1
    u = now[0]
    assert u.direction == -1 and u.known == 60 + S["w_double"] and u.deadline == 60 + 30 - 1
    assert u.target_if(-1, 80) == pytest.approx(85 - (100.5 - 85))  # trough minus the pattern's height

    def status(tail):
        st, _ = _setups_and_signals(_path(shape + tail))
        return [(x.status, x.resolved) for x in st if x.rule == "Double top" and x.known == u.known]
    assert status([(85, 101), (100, 101)]) == [("Failed", 85)]      # closed above the higher peak first
    assert status([(100, 90)]) == [("Expired", 90)]                 # no break before the deadline
    broke = status([(82, 90), (86, 84), (100, 84)])
    assert broke and broke[0][0] == "Broke out"


def test_status_counts_table():
    from atrade.forming import status_counts
    setups, sigs = _setups_and_signals(_path([(0, 80), (30, 100), (45, 85), (60, 100.5), (80, 90)]))
    t = status_counts(setups, sigs)
    row = t.set_index("rule").loc["Double top"]
    assert row["shapes"] >= 1 and row["Forming"] >= 1
