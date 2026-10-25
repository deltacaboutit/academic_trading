"""Tests for the per-stock options store, the Watchtower sync and the GEX add-on."""
import datetime as dt
import subprocess
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
        for e in pd.bdate_range(d, periods=n_exp * 5)[::5]:
            T = (e - pd.Timestamp(d)).days / 365 + 0.001
            for K in np.round(np.linspace(0.8 * spot, 1.2 * spot, 30), 1):
                for c in (True, False):
                    rows.append({"date": d, "ticker": t, "spot": spot, "expiry_str": e.strftime("%Y-%m-%d"),
                                 "strike": K, "is_call": c, "iv": 0.25 + rng.normal(0, 0.02),
                                 "oi": float(rng.integers(1, 3000)), "T": T,
                                 "volume": float(rng.integers(0, 500)) if volume else np.nan, "source": "test"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- clean_iv ---

def test_clean_iv_interpolates_within_same_expiry_and_type():
    df = pd.DataFrame({"date": dt.date(2026, 1, 5), "expiry_str": "2026-02-20", "is_call": True,
                       "strike": [90.0, 95, 100, 105, 110], "iv": [0.40, 0.00001, 0.30, 5.0, 0.20]})
    other = df.assign(is_call=False, iv=[0.5, 0.5, 0.5, 0.5, 0.5])      # puts must not influence calls
    out = OS.clean_iv(pd.concat([df, other], ignore_index=True))
    calls = out[out["is_call"]]
    assert calls["iv"].tolist() == pytest.approx([0.40, 0.35, 0.30, 0.25, 0.20])
    assert calls["iv_filled"].tolist() == [False, True, False, True, False]
    assert not out.loc[~out["is_call"], "iv_filled"].any()


def test_clean_iv_edges_use_nearest_good_value_and_all_junk_stays_missing():
    df = pd.DataFrame({"date": dt.date(2026, 1, 5), "expiry_str": ["A"] * 3 + ["B"] * 2, "is_call": True,
                       "strike": [90.0, 100, 110, 100, 110], "iv": [0.00001, 0.3, 5.0, 0.00001, np.nan]})
    out = OS.clean_iv(df)
    assert out["iv"].iloc[:3].tolist() == pytest.approx([0.3, 0.3, 0.3])
    assert out["iv"].iloc[3:].isna().all()


# ------------------------------------------------------ per-stock files ---

def test_one_file_per_stock_appends_days(tmp_path):
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=5)):
        day = fake_day(d.date(), seed=i)
        for t, g in day.groupby("ticker"):
            OS.append_stock(t, g, chain_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["AAPL.parquet", "SPY.parquet"]
    spy = OS.read_chain("SPY", chain_dir=tmp_path, download=False)
    assert spy["date"].nunique() == 5 and (spy["ticker"] == "SPY").all()
    part = OS.read_chain("SPY", start="2026-01-06", end="2026-01-07", chain_dir=tmp_path, download=False)
    assert part["date"].nunique() == 2


def test_keep_new_replaces_whole_day_and_keep_old_only_fills_gaps(tmp_path):
    d = dt.date(2026, 3, 2)
    full = fake_day(d, tickers=("SPY",), seed=1)
    OS.append_stock("SPY", full, chain_dir=tmp_path)
    small = fake_day(d, tickers=("SPY",), n_exp=1, seed=2)
    OS.append_stock("SPY", small, chain_dir=tmp_path, keep="old")            # day exists -> ignored
    assert len(OS.read_chain("SPY", chain_dir=tmp_path, download=False)) == len(full)
    OS.append_stock("SPY", small, chain_dir=tmp_path, keep="new")            # replaces the whole day
    assert len(OS.read_chain("SPY", chain_dir=tmp_path, download=False)) == len(small)


def test_summary_upsert_monthly_files_and_ratios(tmp_path):
    days = [dt.date(2026, 1, 30), dt.date(2026, 2, 2)]
    ch = pd.concat([fake_day(d, seed=i) for i, d in enumerate(days)])
    s = OS.summarize(OS.normalize_chain(ch, "x"))
    OS.upsert_summary(s, tmp_path)
    assert sorted(f.name for f in (tmp_path / "summary").glob("*.parquet")) == ["2026-01.parquet", "2026-02.parquet"]
    back = OS.read_summary(store=tmp_path)
    r = back.iloc[0]; g = ch[(ch["date"] == r["date"].date()) & (ch["ticker"] == r["ticker"])]
    assert r["pc_oi"] == pytest.approx(g[~g.is_call].oi.sum() / g[g.is_call].oi.sum())
    assert r["pc_volume"] == pytest.approx(g[~g.is_call].volume.sum() / g[g.is_call].volume.sum())
    OS.upsert_summary(s, tmp_path)
    assert len(OS.read_summary(store=tmp_path)) == len(s)
    assert s["ivx_30d"].between(20, 30).all()                # fake chains have IV ~25%


def test_ivx_blends_the_two_expiries_around_30_days():
    rows = []
    for e, d, iv in (("2026-01-12", 7, 0.50), ("2026-01-25", 20, 0.20), ("2026-02-14", 40, 0.40),
                     ("2026-03-20", 74, 0.90)):
        for K in (80.0, 95, 99, 100, 101, 105, 120):
            for c in (True, False):
                rows.append({"expiry_str": e, "T": d / 365, "strike": K, "is_call": c,
                             "iv": 3.0 if K in (80, 120) else iv})          # far strikes must be ignored
    ch = pd.DataFrame(rows)
    w = (np.sqrt(40) - np.sqrt(30)) / (np.sqrt(40) - np.sqrt(20))
    assert OS.ivx(ch, 100.0) == pytest.approx(100 * (w * 0.20 + (1 - w) * 0.40))
    assert OS.ivx(ch[ch["T"] > 50 / 365], 100.0) == pytest.approx(90.0)       # no extrapolation
    assert np.isnan(OS.ivx(ch.assign(iv=np.nan), 100.0))


# ------------------------------------------------------------------ GEX ---

def test_gex_levels_simple_case():
    ch = pd.DataFrame({"strike": [110.0, 90.0], "is_call": [True, False], "iv": [0.3, 0.3],
                       "oi": [1000.0, 1000.0], "T": [0.1, 0.1]})
    lv = G.levels(ch, 100.0, scale=1.0)
    assert lv["call_wall_price"] == 110 and lv["put_wall_price"] == 90
    assert 95 < lv["gex_flip"] < 105


def test_gex_signals_use_previous_day_levels():
    idx = pd.bdate_range("2026-01-05", periods=5)
    df = pd.DataFrame({"Open": 100.0, "High": [101, 101, 101, 101, 101.0], "Low": [99, 99, 94.5, 99, 99.0],
                       "Close": [100.0, 100, 96, 100, 100]}, index=idx)
    lv = {"put_wall_price": np.array([95, 95, 97, 95, 95.0]), "call_wall_price": np.full(5, 120.0),
          "gex_flip": np.full(5, 80.0)}
    sig = [s for s in G.gex_signals(df, lv) if s.rule == "[GEX] Put-wall hold"]
    assert [s.idx for s in sig] == [2]


# --------------------------------------------------------- Watchtower sync ---

def _wt_day(root: Path, label: str, session: dt.date, tickers=("SPY", "AAPL"), n_exp=4, last_hour_utc=20):
    feats = []
    for t in tickers:
        c = fake_day(session, tickers=(t,), n_exp=n_exp).drop(columns=["date", "ticker", "spot", "source"])
        c["lastTradeDate"] = pd.Timestamp(session).tz_localize("UTC") + pd.Timedelta(hours=last_hour_utc)
        c.loc[c.index[:3], "iv"] = 0.00001
        p = root / "data" / "chains" / f"ticker={t}"; p.mkdir(parents=True, exist_ok=True)
        c.to_parquet(p / f"date={label}.parquet")
        feats.append({"ticker": t, "spot": 123.0, "gex_total": 1.0, "flip_level": 120.0, "call_wall": 130.0,
                      "call_wall_gex": 5.0, "put_wall": 110.0, "put_wall_gex": -5.0, "ivx_30d": 20.0})
    p = root / "data" / "features"; p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feats).to_parquet(p / f"date={label}.parquet")


def test_watchtower_sync_fixes_dates_skips_partial_runs_and_is_incremental(tmp_path, monkeypatch):
    from atrade import watchtower as W
    src = tmp_path / "wt"
    run = lambda *a: subprocess.run(["git", "-C", str(src), *a], check=True, capture_output=True)
    src.mkdir()
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    run("config", "uploadpack.allowFilter", "true")
    # normal day; a late run labelled the next UTC day; three more full days; a broken partial run
    _wt_day(src, "2026-10-13", dt.date(2026, 10, 13))
    _wt_day(src, "2026-10-15", dt.date(2026, 10, 14))           # started after 8 pm NY on the 14th
    _wt_day(src, "2026-10-16", dt.date(2026, 10, 16))
    _wt_day(src, "2026-10-17", dt.date(2026, 10, 16), tickers=("SPY",), n_exp=1)   # Saturday partial re-run
    run("add", "-A"); run("commit", "-qm", "data")
    monkeypatch.setattr(W, "WT_URL", src.as_uri())
    monkeypatch.setattr(W, "MIN_CONTRACTS", 300)
    store, chains = tmp_path / "store", tmp_path / "chains"
    r = W.sync(tmp_path / "work", store=store, chain_dir=chains, log=lambda *_: None)
    assert r["labels"] == 4
    spy = OS.read_chain("SPY", chain_dir=chains, download=False)
    assert sorted(map(str, spy["date"].unique())) == ["2026-10-13", "2026-10-14", "2026-10-16"]
    assert len(spy[spy["date"] == dt.date(2026, 10, 16)]) == 4 * 30 * 2       # full run kept, partial ignored
    assert spy["iv"].between(0.01, 5).all() and spy["iv_filled"].sum() == 9
    s = OS.read_summary(store=store)
    assert (s["gex_source"] == "watchtower").all() and s["gex_flip"].eq(120).all()
    # second run: nothing new
    assert W.sync(tmp_path / "work", store=store, chain_dir=chains, log=lambda *_: None)["labels"] == 0
    # a new day appears in Watchtower -> only that day is processed
    _wt_day(src, "2026-10-19", dt.date(2026, 10, 19)); run("add", "-A"); run("commit", "-qm", "more")
    r = W.sync(tmp_path / "work", store=store, chain_dir=chains, log=lambda *_: None)
    assert r["labels"] == 1
    assert OS.read_chain("AAPL", chain_dir=chains, download=False)["date"].nunique() == 4


def test_import_fills_only_missing_days(tmp_path):
    from atrade import import_history as IH
    store, chains = tmp_path / "store", tmp_path / "chains"
    d1, d2 = dt.date(2026, 5, 4), dt.date(2026, 5, 5)
    existing = OS.normalize_chain(fake_day(d1, tickers=("SPY",), seed=1), "watchtower")
    OS.append_stock("SPY", existing, chain_dir=chains)
    OS.upsert_summary(OS.summarize(existing), store)
    flat = pd.concat([fake_day(d1, tickers=("SPY",), seed=9), fake_day(d2, tickers=("SPY",), seed=9)])
    flat.drop(columns=["volume"]).to_parquet(tmp_path / "chain_history.parquet")
    IH.main(["--chain", str(tmp_path / "chain_history.parquet")], store=store, chain_dir=chains, log=lambda *_: None)
    spy = OS.read_chain("SPY", chain_dir=chains, download=False)
    kept = spy[spy["date"] == d1].sort_values(["expiry_str", "is_call", "strike"])["oi"].to_numpy()
    assert np.array_equal(kept, existing.sort_values(["expiry_str", "is_call", "strike"])["oi"].to_numpy())
    assert spy["date"].nunique() == 2


# ------------------------------------------------------ whole pipeline ---

def test_full_analysis_reads_store_substitutes_and_gex(tmp_path, monkeypatch):
    from atrade import data as D
    from atrade.analysis import analyze
    from atrade.config import DEFAULTS
    from atrade.demo import demo_prices
    monkeypatch.setattr(OS, "STORE", tmp_path)
    monkeypatch.setattr(D, "load_vix", lambda years: None)
    px = demo_prices()
    rng = np.random.default_rng(3)
    rows = []
    for i, d in enumerate(px.index[-160:]):
        c = float(px.loc[d, "Close"])
        for t in ("SPY", "AAPL"):
            r = {"date": d.date(), "ticker": t, "spot": c, "call_oi": 1e5, "put_oi": 1e5 * (0.8 + 0.4 * rng.random()),
                 "avg_iv": 0.2 + 0.05 * rng.random(), "gex_flip": c * 0.98, "call_wall_price": c * 1.03,
                 "put_wall_price": c * 0.97, "gex_source": "watchtower", "source": "watchtower"}
            r["pc_oi"] = r["put_oi"] / r["call_oi"]
            if i >= 150:
                r.update({"call_volume": 5e4, "put_volume": 4e4, "pc_volume": 0.8})
            rows.append(r)
    OS.upsert_summary(pd.DataFrame(rows), tmp_path)
    r = analyze("SPY", dict(DEFAULTS), prices=px)
    st = pd.DataFrame(r["sentiment_status"])
    assert st.loc[st["system"].str.startswith("Tsunami Sentiment Wave - open"), "votes"].iloc[0]
    assert r["gex"] is not None and len(r["gex_signals"]) > 0
    assert not any(sg.rule.startswith("[GEX]") for sg, c, _ in r["votes"])
    assert r["results"]["rule"].str.startswith("[GEX]").any()


def test_inbox_zip_of_missing_days_is_imported(tmp_path):
    import zipfile
    from atrade import import_history as IH
    wt = tmp_path / "missing"
    _wt_day(wt, "2026-08-20", dt.date(2026, 8, 20))
    inbox = tmp_path / "inbox"; inbox.mkdir()
    with zipfile.ZipFile(inbox / "missing_days.zip", "w") as z:
        for f in (wt / "data").rglob("*.parquet"):
            z.write(f, f.relative_to(wt / "data"))          # zip holds chains/ and features/ directly
    store, chains = tmp_path / "store", tmp_path / "chains"
    IH.main(["--inbox", str(inbox)], store=store, chain_dir=chains, log=lambda *_: None)
    assert OS.read_chain("AAPL", chain_dir=chains, download=False)["date"].unique().tolist() == [dt.date(2026, 8, 20)]
    assert len(OS.read_summary(store=store)) == 2


def test_release_bundles_round_trip(tmp_path):
    from atrade import release as R
    src, out, back = tmp_path / "src", tmp_path / "zips", tmp_path / "back"
    for t in ("AAPL", "ABBV", "BRK.B", "SPY"):
        OS.append_stock(t, fake_day(dt.date(2026, 1, 5), tickers=(t,)), chain_dir=src)
    assert R.shard_of("AAPL") == "A" and R.shard_of("^VIX") == "0"
    z = R.pack_shard(src, "A", out)
    assert sorted(R.unpack(z, back)) == ["AAPL.parquet", "ABBV.parquet"]
    assert (back / "AAPL.parquet").read_bytes() == (src / "AAPL.parquet").read_bytes()
