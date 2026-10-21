"""Daily option-chain collector, tested against a stand-in for Yahoo Finance built from stored chains."""
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atrade import collect_chains as CC
from atrade import gex as G
from atrade import options_store as OS

SESSION = dt.date(2026, 10, 19)


def yahoo_like(n_exp=4, spot=100.0, seed=0):
    """Calls and puts in Yahoo's column layout for a few expiries."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n_exp):
        e = (SESSION + dt.timedelta(days=4 + 7 * k)).isoformat()
        K = np.arange(70, 131, 5.0)
        mk = lambda: pd.DataFrame({"contractSymbol": "X", "strike": K, "lastPrice": 1.0, "bid": 0.9, "ask": 1.1,
                                   "volume": rng.integers(0, 500, len(K)).astype(float),
                                   "openInterest": rng.integers(0, 3000, len(K)),
                                   "impliedVolatility": np.r_[1e-5, np.full(len(K) - 2, 0.3), 0.35]})
        out[e] = SimpleNamespace(calls=mk(), puts=mk())
    out[(SESSION + dt.timedelta(days=200)).isoformat()] = None        # beyond 90 days: must not be fetched
    return out


class FakeYF:
    def __init__(self, chains, fail=(), stale=()):
        self.chains, self.fail, self.stale, self.calls = chains, set(fail), set(stale), []

    def Ticker(self, t):
        yf = self
        class T:
            options = tuple(sorted(yf.chains))
            def history(self, period=None, auto_adjust=False):
                if t in yf.fail:
                    raise ConnectionError("Too Many Requests")
                last = SESSION - dt.timedelta(days=1) if t in yf.stale else SESSION
                idx = pd.bdate_range(end=last, periods=5)
                return pd.DataFrame({"Close": np.linspace(96, 100, 5)}, index=idx)
            def option_chain(self, e):
                yf.calls.append((t, e))
                assert yf.chains[e] is not None, "fetched an expiry beyond 90 days"
                return yf.chains[e]
        return T()


def test_session_day_waits_for_the_close():
    ny = CC.NY
    assert CC.session_day(dt.datetime(2026, 10, 21, 15, 0, tzinfo=ny)) == dt.date(2026, 10, 20)
    assert CC.session_day(dt.datetime(2026, 10, 21, 16, 30, tzinfo=ny)) == dt.date(2026, 10, 21)
    assert CC.session_day(dt.datetime(2026, 10, 24, 12, 0, tzinfo=ny)) == dt.date(2026, 10, 23)   # Saturday
    assert CC.session_day(dt.datetime(2026, 11, 26, 18, 0, tzinfo=ny)) == dt.date(2026, 11, 25)   # Thanksgiving


def test_years_to_expiry_counts_from_the_session_close():
    assert CC.years_to_expiry("2026-10-20", SESSION) == pytest.approx(1 / 365)


def test_collect_saves_chain_summary_and_skips_done(tmp_path, monkeypatch):
    monkeypatch.setattr(CC, "risk_free_rate", lambda s, store=None: 0.04)
    yf = FakeYF(yahoo_like(), fail={"BAD"}, stale={"LATE"})
    log = []
    r = CC.collect(["AAA", "BAD", "LATE"], SESSION, store=tmp_path, chain_dir=tmp_path / "chains", yf=yf,
                   sleep=lambda s: None, log=log.append)
    assert r["ok"] == 1 and set(r["failed"]) == {"BAD", "LATE"} and "not 2026-10-19 yet" in r["failed"]["LATE"]
    ch = OS.read_chain("AAA", chain_dir=tmp_path / "chains", download=False)
    assert (ch["oi"] >= 5).all() and (ch["date"] == SESSION).all() and (ch["source"] == "yahoo").all()
    assert ch["expiry_str"].nunique() == 4                                  # the 200-day expiry was not fetched
    assert ch["iv_filled"].any() and (ch["iv"] > 0.01).all()                # Yahoo's 0.00001 IV was filled
    s = OS.read_summary(store=tmp_path)
    assert list(s["ticker"]) == ["AAA"] and s["spot"].iloc[0] == 100 and np.isfinite(s["gex_total"].iloc[0])
    assert s["gex_source"].iloc[0] == "computed" and np.isfinite(s["pc_volume"].iloc[0])
    st = __import__("json").loads((tmp_path / "collect_state.json").read_text())
    assert st[str(SESSION)]["collected"] == 1 and "BAD" in st[str(SESSION)]["failed"]
    # a catch-up run only fetches what is missing
    yf2 = FakeYF(yahoo_like())
    r2 = CC.collect(["AAA", "BAD", "LATE"], SESSION, store=tmp_path, chain_dir=tmp_path / "chains", yf=yf2,
                    sleep=lambda s: None, log=log.append)
    assert r2["todo"] == 2 and r2["ok"] == 2 and {t for t, _ in yf2.calls} == {"BAD", "LATE"}
    assert CC.collect(["AAA", "BAD", "LATE"], SESSION, store=tmp_path, yf=yf2, log=log.append)["todo"] == 0


def test_walls_are_strikes_and_flip_from_the_profile():
    """Walls sit on real strikes (largest call GEX / most negative put GEX); the flip lies between them."""
    K = np.r_[np.arange(80, 100, 5.0), np.arange(105, 125, 5.0)]
    oi = np.where(K == 110, 9000, np.where(K == 90, 7000, 1000.0))
    c = pd.DataFrame({"strike": K, "is_call": K > 100, "iv": 0.3, "oi": oi, "T": 20 / 365})
    lv = G.daily_levels(c, 100.0, 0.04)
    assert lv["call_wall_price"] == 110 and lv["put_wall_price"] == 90 and lv["wall_method"] == "strike"
    assert lv["call_wall_gex"] > 0 > lv["put_wall_gex"]
    assert 90 < lv["gex_flip"] < 110


def test_no_collection_while_the_market_is_open():
    ny = CC.NY
    assert CC.market_open_now(dt.datetime(2026, 10, 21, 11, 0, tzinfo=ny))
    assert not CC.market_open_now(dt.datetime(2026, 10, 21, 8, 30, tzinfo=ny))
    assert not CC.market_open_now(dt.datetime(2026, 10, 21, 16, 30, tzinfo=ny))
    assert not CC.market_open_now(dt.datetime(2026, 10, 24, 11, 0, tzinfo=ny))     # Saturday
