"""Tests for the options data store, the daily update and the GEX add-on."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade import gex as G
from atrade import options_store as OS


def fake_day(d, tickers=("SPY", "AAPL"), n_exp=4, seed=0, volume=True):
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        spot = 100 + 50 * rng.random()
        for k, e in enumerate(pd.bdate_range(d, periods=n_exp * 5)[::5]):
            T = (e - pd.Timestamp(d)).days / 365 + 0.001
            for K in np.round(np.linspace(0.8 * spot, 1.2 * spot, 30), 1):
                for c in (True, False):
                    rows.append({"date": d, "ticker": t, "spot": spot, "expiry_str": e.strftime("%Y-%m-%d"),
                                 "strike": K, "is_call": c, "iv": 0.25 + rng.normal(0, 0.02),
                                 "oi": float(rng.integers(1, 3000)), "T": T,
                                 "volume": float(rng.integers(0, 500)) if volume else np.nan, "source": "yahoo"})
    return pd.DataFrame(rows)


def test_parts_stay_under_limit_and_rebuild_exactly(tmp_path):
    days = pd.bdate_range("2026-01-05", periods=12)
    hist = pd.concat([fake_day(d.date(), seed=i) for i, d in enumerate(days)], ignore_index=True)
    limit = 30_000       # bytes - tiny on purpose so the history must split
    parts = OS.write_history_parts(hist, tmp_path, max_bytes=limit)
    assert len(parts) > 1
    assert all(p["bytes"] <= limit or p["first"] == p["last"] for p in parts)
    back = OS.read_chain(store=tmp_path)
    assert len(back) == len(hist)
    a = OS.normalize_chain(hist, "x").drop(columns="source").reset_index(drop=True)
    b = OS.normalize_chain(back, "x").drop(columns="source").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_pending_days_compact_into_new_part_without_rewriting_old_parts(tmp_path):
    OS.write_history_parts(fake_day(dt.date(2026, 1, 2), seed=99), tmp_path, max_bytes=10**9)
    first = (tmp_path / "chain" / "part_001.parquet").read_bytes()
    limit = 40_000
    compacted = False
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=10)):
        r = OS.add_chain_day(fake_day(d.date(), seed=i), tmp_path, max_bytes=limit)
        compacted |= r["compacted"]
    assert compacted
    assert (tmp_path / "chain" / "part_001.parquet").read_bytes() == first      # sealed part untouched
    full = OS.read_chain(store=tmp_path)
    assert full["date"].nunique() == 11
    assert len(OS.read_chain(tickers=["SPY"], start="2026-01-06", end="2026-01-07", store=tmp_path)["date"].unique()) == 2


def test_rerunning_a_day_replaces_it(tmp_path):
    d = dt.date(2026, 3, 2)
    OS.add_chain_day(fake_day(d, seed=1), tmp_path)
    OS.add_chain_day(fake_day(d, seed=2), tmp_path)
    assert len(OS.read_chain(store=tmp_path)) == len(fake_day(d))


def test_summary_upsert_monthly_files_and_ratios(tmp_path):
    days = [dt.date(2026, 1, 30), dt.date(2026, 2, 2)]
    ch = pd.concat([fake_day(d, seed=i) for i, d in enumerate(days)])
    s = OS.summarize(ch)
    OS.upsert_summary(s, tmp_path)
    assert sorted(f.name for f in (tmp_path / "summary").glob("*.parquet")) == ["2026-01.parquet", "2026-02.parquet"]
    back = OS.read_summary(store=tmp_path)
    r = back.iloc[0]; g = ch[(ch["date"] == r["date"].date()) & (ch["ticker"] == r["ticker"])]
    assert r["pc_oi"] == pytest.approx(g[~g.is_call].oi.sum() / g[g.is_call].oi.sum())
    assert r["pc_volume"] == pytest.approx(g[~g.is_call].volume.sum() / g[g.is_call].volume.sum())
    # re-upserting the same rows does not duplicate them
    OS.upsert_summary(s, tmp_path)
    assert len(OS.read_summary(store=tmp_path)) == len(s)


def test_avg_iv_ignores_junk_values():
    ch = fake_day(dt.date(2026, 1, 5), tickers=("SPY",))
    clean_mean = ch[ch.volume > 0].iv.mean()
    ch.loc[ch.index[:5], ["iv", "volume"]] = [107.0, 10]
    assert OS.summarize(ch, compute_gex=False)["avg_iv"].iloc[0] == pytest.approx(clean_mean, abs=0.01)


def test_gex_levels_simple_case():
    # one big call at 110 and one big put at 90: call wall 110, put wall 90, flip in between
    ch = pd.DataFrame({"strike": [110.0, 90.0], "is_call": [True, False], "iv": [0.3, 0.3],
                       "oi": [1000.0, 1000.0], "T": [0.1, 0.1]})
    lv = G.levels(ch, 100.0, scale=1.0)
    assert lv["call_wall_price"] == 110 and lv["put_wall_price"] == 90
    assert 95 < lv["gex_flip"] < 105
    g = G.gamma(100, 110, 0.1, 0.3) * 1000 * 100 * 100 ** 2
    assert lv["call_wall_gex"] == pytest.approx(g)


def test_gex_signals_use_previous_day_levels():
    idx = pd.bdate_range("2026-01-05", periods=5)
    df = pd.DataFrame({"Open": 100.0, "High": [101, 101, 101, 101, 101.0], "Low": [99, 99, 94.5, 99, 99.0],
                       "Close": [100.0, 100, 96, 100, 100]}, index=idx)
    lv = {"put_wall_price": np.array([95, 95, 97, 95, 95.0]), "call_wall_price": np.full(5, 120.0),
          "gex_flip": np.full(5, 80.0)}
    sig = [s for s in G.gex_signals(df, lv) if s.rule == "[GEX] Put-wall hold"]
    # day 2: low 94.5 <= yesterday's wall 95 and close 96 > 95 -> buy; today's own level (97) is NOT used
    assert [s.idx for s in sig] == [2]


def test_update_runs_offline_with_fake_yahoo(tmp_path, monkeypatch):
    from atrade import update_options as U
    monkeypatch.setattr(OS, "STORE", tmp_path)
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "universe.txt").write_text("SPY\nAAPL\nBAD\n")

    def fake_fetch(t, now, tries=3):
        if t == "BAD":
            raise RuntimeError("no listed options")
        df = fake_day(now.date(), tickers=(t,))
        df.attrs["bar_day"] = now.date()
        return df
    monkeypatch.setattr(U, "fetch_one", fake_fetch)
    assert U.main(["--force"]) == 0
    s = OS.read_summary(store=tmp_path)
    assert set(s["ticker"]) == {"SPY", "AAPL"} and s["pc_volume"].notna().all() and s["gex_flip"].notna().any()
    assert (tmp_path / "update_log.csv").exists()


def test_full_analysis_reads_store_substitutes_and_gex(tmp_path, monkeypatch):
    """The whole pipeline with a fake store: open-interest substitutes vote while
    real volume is short, and GEX signals are back-tested but never vote."""
    from atrade import data as D
    from atrade.analysis import analyze
    from atrade.config import DEFAULTS
    from atrade.demo import demo_prices
    monkeypatch.setattr(OS, "STORE", tmp_path)
    monkeypatch.setattr(D, "load_vix", lambda years: None)
    px = demo_prices()
    dates = px.index[-160:]
    rng = np.random.default_rng(3)
    rows = []
    for i, d in enumerate(dates):
        c = float(px.loc[d, "Close"])
        for t in ("SPY", "AAPL"):
            rows.append({"date": d.date(), "ticker": t, "spot": c, "call_oi": 1e5, "put_oi": 1e5 * (0.8 + 0.4 * rng.random()),
                         "pc_oi": np.nan, "call_volume": np.nan, "put_volume": np.nan, "pc_volume": np.nan,
                         "avg_iv": 0.2 + 0.05 * rng.random(), "n_contracts": 100, "gex_total": 1e9,
                         "gex_flip": c * 0.98, "call_wall_price": c * 1.03, "call_wall_gex": 1e9,
                         "put_wall_price": c * 0.97, "put_wall_gex": -1e9, "gex_source": "gex_history", "source": "import"})
            rows[-1]["pc_oi"] = rows[-1]["put_oi"] / rows[-1]["call_oi"]
            if i >= 150:   # only 10 days of real volume
                rows[-1].update({"call_volume": 5e4, "put_volume": 4e4, "pc_volume": 0.8})
    OS.upsert_summary(pd.DataFrame(rows)[OS.SUMMARY_COLS], tmp_path)
    r = analyze("SPY", dict(DEFAULTS), prices=px)
    st = pd.DataFrame(r["sentiment_status"])
    assert st.loc[st["system"].str.startswith("Tsunami Sentiment Wave - open"), "votes"].iloc[0]
    assert not st.loc[st["system"] == "Tsunami Sentiment Wave", "active"].iloc[0]
    assert r["gex"] is not None and len(r["gex_signals"]) > 0
    voted = {sg.rule for sg, c, _ in r["votes"]}
    assert not any(x.startswith("[GEX]") for x in voted)
    assert r["results"]["rule"].str.startswith("[GEX]").any()
