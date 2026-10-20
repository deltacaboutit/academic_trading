"""Version 2 checks: Framework 9 bootstrap and tests, [LMW] items, DTW similarity, research runner."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade import bootstrap as BS
from atrade import lmw
from atrade import similarity as SIM
from atrade import validation as V
from atrade.config import DEFAULTS
from atrade.signals import Signal

S = dict(DEFAULTS)


def garch_returns(n=4000, a0=0.05, a1=0.08, beta=0.9, seed=0):
    rng = np.random.default_rng(seed)
    h = a0 / (1 - a1 - beta); e_prev = 0.0; out = np.empty(n)
    for t in range(n):
        h = a0 + a1 * e_prev ** 2 + beta * h
        e_prev = math.sqrt(h) * rng.standard_normal()
        out[t] = 0.03 + e_prev
    return out


# ------------------------------------------------------------ step 3 -----

def test_iid_tests_accept_noise_and_reject_volatility_clusters():
    noise = np.random.default_rng(1).standard_normal(3000)
    assert BS.ljung_box(noise) > 0.10 and BS.arch_test(noise) > 0.10
    assert BS.arch_test(garch_returns()) < 0.01


def test_garch_m_recovers_its_parameters_and_qualifies():
    r = garch_returns(5000, seed=3)
    close = 50 * np.exp(np.concatenate([[0], np.cumsum(r / 100)]))
    fits = {f["model"]: f for f in BS.fit_models(close)}
    g = fits["GARCH-M"]
    assert g["coef"]["a1"] == pytest.approx(0.08, abs=0.03) and g["coef"]["beta"] == pytest.approx(0.90, abs=0.04)
    assert g["qualified"]
    assert not fits["AR(1)"]["qualified"]                      # volatility clusters left in AR(1) residuals


def test_simulated_histories_keep_start_and_length():
    r = garch_returns(1500, seed=5)
    close = 20 * np.exp(np.concatenate([[0], np.cumsum(r / 100)]))
    m = [f for f in BS.fit_models(close) if f["model"] == "GARCH-M"][0]["obj"]
    sims = list(BS.simulate_prices(m, close, 3, seed=1))
    assert all(len(x) == len(close) and x[0] == pytest.approx(close[0]) for x in sims)
    assert not np.allclose(sims[0], sims[1])


def test_bootstrap_verdict_is_strict_across_models():
    res = pd.DataFrame([
        {"rule": "A", "h": 10, "block": "Final test", "model": "GARCH-M", "p_adj": 0.01},
        {"rule": "A", "h": 10, "block": "Final test", "model": "E-GARCH", "p_adj": 0.02},
        {"rule": "B", "h": 10, "block": "Final test", "model": "GARCH-M", "p_adj": 0.01},
        {"rule": "B", "h": 10, "block": "Final test", "model": "E-GARCH", "p_adj": 0.30}])
    v = BS.verdict_table({"testable": True, "results": res}, S).set_index("rule")["bootstrap"]
    assert v["A"] == "passed" and v["B"] == "not passed"
    nt = BS.verdict_table({"testable": False, "rules": ["A"]}, S)
    assert nt["bootstrap"].iloc[0] == "not testable"


# ------------------------------------------------------- steps 6 and 8 -----

def test_reality_check_detects_a_real_edge_and_not_noise():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1000, 20)))
    assert V.reality_check(noise, B=300)["rc_p"] > 0.10
    edge = noise.copy(); edge[0] += 0.003
    out = V.reality_check(edge, B=300)
    assert out["rc_p"] < 0.05 and out["spa_p"] < 0.05 and out["best_rule"] == 0


def test_forecast_accuracy_perfect_and_naive():
    close = np.linspace(100, 130, 200)
    sigs = [Signal("Double bottom", "reversal", "pattern", 1, i, target=close[i + 10], time_limit=i + 10)
            for i in range(0, 150, 15)]
    fc = V.forecast_accuracy(sigs, close).set_index("pattern")
    assert fc.loc["All patterns", "MSE"] == pytest.approx(0, abs=1e-12)
    assert fc.loc["All patterns", "POCID %"] == 100
    nochange = [Signal("Double bottom", "reversal", "pattern", 1, i, target=close[i], time_limit=i + 10)
                for i in range(0, 150, 15)]
    fc2 = V.forecast_accuracy(nochange, close).set_index("pattern")
    assert fc2.loc["All patterns", "NPRMSE"] == pytest.approx(1.0)       # 'no change' forecast scores exactly 1


def test_rating_performance_long_only_matches_buy_and_hold():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.01, 500)))
    perf = V.rating_performance(np.ones(500, int), close, 0, 500, 0.0)
    assert perf["total_return"] == pytest.approx(perf["buy_and_hold"])      # always long = buy-and-hold
    short = V.rating_performance(-np.ones(500, int), close, 0, 500, 0.0)
    assert short["total_return"] == pytest.approx(-perf["buy_and_hold"])


# ----------------------------------------------------------------- LMW -----

def _ext(vals, start="P"):
    kinds = ["P", "T"] if start == "P" else ["T", "P"]
    return [(i * 6, float(v), kinds[i % 2]) for i, v in enumerate(vals)]


@pytest.mark.parametrize("vals,start,name", [
    ([100, 90, 110, 90.5, 100.5], "P", "HS-LMW"),
    ([100, 110, 90, 109.5, 99.5], "T", "IHS-LMW"),
    ([100, 95, 104, 90, 108], "P", "BTOP"),
    ([108, 90, 104, 95, 100], "P", "TTOP"),
    ([100, 95, 100.3, 95.2, 99.8], "P", "RTOP"),
])
def test_lmw_shapes(vals, start, name):
    assert name in lmw.classify(_ext(vals, start))


def test_lmw_double_top_needs_22_days():
    ext = [(0, 100.0, "P"), (12, 92.0, "T"), (25, 100.5, "P"), (34, 95.0, "T")]
    assert "DTOP-LMW" in lmw.classify(ext)
    ext2 = [(0, 100.0, "P"), (12, 92.0, "T"), (20, 100.5, "P"), (34, 95.0, "T")]
    assert "DTOP-LMW" not in lmw.classify(ext2)


def test_lmw_no_look_ahead():
    from atrade.demo import demo_prices
    c = demo_prices()["Close"].to_numpy(float)
    full = lmw.detect(c, None)
    cut = 2500
    part = lmw.detect(c[:cut + 1], None)
    a = full[full["detect_idx"] <= cut][["pattern", "detect_idx"]].reset_index(drop=True)
    b = part[["pattern", "detect_idx"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_lmw_information_tests_on_same_distribution():
    rng = np.random.default_rng(9)
    r = lmw.info_tests(rng.standard_normal(400), rng.standard_normal(20000))
    assert r["p_decile"] > 0.01 and r["p_ks"] > 0.01
    shifted = lmw.info_tests(rng.standard_normal(400) + 0.5, rng.standard_normal(20000))
    assert shifted["p_ks"] < 0.001


# ----------------------------------------------------------------- DTW -----

def test_dtw_finds_a_planted_copy():
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 1, 1500)) + 200
    shape = np.cumsum(rng.normal(0, 2, 61))
    base[400:461] = base[399] + shape - shape[0]
    base[-61:] = base[-62] + shape - shape[0]
    out = SIM.find_matches(base, 61, tau=15)
    best = out["matches"].iloc[0]
    # the 21-day smoothing blends the copy's last days with what follows it, so allow half a window
    assert 400 - 11 <= best["start_idx"] <= 400 + 11 and 460 - 11 <= best["end_idx"] <= 460 + 11
    assert best["cost"] < 15 and (out["matches"]["cost"].iloc[1:] > best["cost"]).all()


def test_dtw_band_limits_path_shape():
    Q = np.sin(np.linspace(0, 6, 40))
    Y = np.concatenate([np.zeros(50), np.sin(np.linspace(0, 6, 80)), np.zeros(50)])   # same shape, twice as slow
    free, _ = SIM.subsequence_dtw(Q, Y)
    banded, _ = SIM.subsequence_dtw(Q, Y, band=5)
    assert np.nanmin(banded) > np.nanmin(free)


# ------------------------------------------------------------- research -----

def test_vectorised_confirmation_matches_the_rating_engine():
    from atrade.research import _confirm
    rng = np.random.default_rng(0)
    for _ in range(100):
        raw = rng.choice([-1, 0, 1], size=200, p=[0.2, 0.6, 0.2]); k = int(rng.integers(1, 4))
        off = np.zeros(200, int); cur = 0; run = 0
        for t in range(200):
            run = run + 1 if t > 0 and raw[t] == raw[t - 1] else 1
            if raw[t] != cur and run >= k:
                cur = raw[t]
            off[t] = cur
        assert (off == _confirm(raw, k)).all()


def test_tuned_weights_switch():
    from atrade.analysis import apply_tuned_weights
    s = dict(S); s["use_tuned_weights"] = False
    assert apply_tuned_weights(s)["w_trend"] == S["w_trend"]
