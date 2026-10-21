"""The consolidated options history.

Two parts:

1. Full contract-level chain - ONE parquet file PER STOCK (SPY.parquet,
   AAPL.parquet, ...) with every day appended. These files live on the GitHub
   Release "options-data" in this repo, not in git history (git would store a
   full new copy of every file every day; a Release file is simply replaced).
   On the Release they are grouped into 27 zip bundles by first letter
   (chains_A.zip ...), see release.py. The tool keeps the files in
   data/chains/ and downloads a stock's bundle the first time it is needed
   each day.

2. Daily summary - one row per stock per day, kept in git as one small file
   per month in options_data/summary/ (put/call ratios, average IV, spot,
   GEX levels). This is what the rating engine reads.

Chain columns: date, ticker, spot, expiry_str, strike, is_call, iv, iv_filled,
oi, volume, T, bid, ask, last_price, source
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import gex as G
from .config import DATA_DIR, ROOT

STORE = ROOT / "options_data"
CHAIN_DIR = DATA_DIR / "chains"
REPO = "deltacaboutit/academic_trading"
RELEASE_TAG = "options-data"
INDEX_ETFS = {"SPY", "QQQ", "IWM", "DIA", "SPX", "^SPX", "XSP", "OEX"}
IV_JUNK_LOW, IV_JUNK_HIGH = 0.01, 4.99

CHAIN_SCHEMA = pa.schema([
    ("date", pa.date32()), ("ticker", pa.string()), ("spot", pa.float64()), ("expiry_str", pa.string()),
    ("strike", pa.float64()), ("is_call", pa.bool_()), ("iv", pa.float64()), ("iv_filled", pa.bool_()),
    ("oi", pa.float64()), ("volume", pa.float64()), ("T", pa.float64()), ("bid", pa.float64()),
    ("ask", pa.float64()), ("last_price", pa.float64()), ("source", pa.string()),
])
CHAIN_COLS = [f.name for f in CHAIN_SCHEMA]
KEY = ["date", "expiry_str", "strike", "is_call"]
SUMMARY_COLS = ["date", "ticker", "spot", "call_oi", "put_oi", "pc_oi", "call_volume", "put_volume", "pc_volume",
                "avg_iv", "ivx_30d", "n_contracts", "gex_total", "gex_flip", "call_wall_price", "call_wall_gex",
                "put_wall_price", "put_wall_gex", "wall_method", "gex_source", "source"]


def paths(store: Path | None = None, chain_dir: Path | None = None):
    s = Path(store) if store else STORE
    return {"root": s, "summary": s / "summary", "chains": Path(chain_dir) if chain_dir else CHAIN_DIR,
            "state": s / "watchtower_state.json", "universe": s / "universe.txt", "manifest": s / "manifest.json"}


def asset_name(ticker: str) -> str:
    return f"{ticker.upper().replace('/', '-')}.parquet"


def release_url(ticker: str) -> str:
    """URL of the zip bundle on the Release that holds this stock's file."""
    c = ticker.upper()[:1]
    shard = c if "A" <= c <= "Z" else "0"
    return f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/chains_{shard}.zip"


# ----------------------------------------------------------- cleaning IV ---

def clean_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Replace junk implied volatilities (Yahoo's 0.00001 floor, its 5.0 cap,
    or missing) by straight-line interpolation between the nearest good strikes
    of the SAME expiry and the SAME type (call or put). Beyond the first or last
    good strike, the nearest good value is used. iv_filled marks changed rows."""
    df = df.copy()
    iv = pd.to_numeric(df["iv"], errors="coerce").to_numpy(float)
    bad = ~((iv > IV_JUNK_LOW) & (iv < IV_JUNK_HIGH))
    out = iv.copy()
    if bad.any():
        grp = df.groupby(["date", "expiry_str", "is_call"], sort=False).indices
        strikes = df["strike"].to_numpy(float)
        for _, ix in grp.items():
            b = bad[ix]
            if not b.any():
                continue
            good = ix[~b]
            if len(good) == 0:
                out[ix] = np.nan
                continue
            order = np.argsort(strikes[good])
            out[ix[b]] = np.interp(strikes[ix[b]], strikes[good][order], iv[good][order])
    df["iv"] = out
    df["iv_filled"] = bad & ~np.isnan(out)
    return df


def normalize_chain(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df.attrs = {}
    df = df.rename(columns={"lastPrice": "last_price"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("spot", "iv", "oi", "volume", "T", "bid", "ask", "last_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float) if col in df else np.nan
    if "expiry_str" not in df:
        df["expiry_str"] = None
    df["expiry_str"] = df["expiry_str"].astype("string").where(df["expiry_str"].notna(), None)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype(float)
    df["is_call"] = df["is_call"].astype(bool)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if "iv_filled" not in df:
        df["iv_filled"] = False
    df["iv_filled"] = df["iv_filled"].fillna(False).astype(bool)
    if "source" not in df:
        df["source"] = source
    df = df[CHAIN_COLS]
    return df.sort_values(["date", "expiry_str", "is_call", "strike"], na_position="first").reset_index(drop=True)


# ------------------------------------------------------ per-stock files ---

def _write(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pandas(df, schema=CHAIN_SCHEMA, preserve_index=False), tmp,
                   compression="zstd", compression_level=9, row_group_size=200_000)
    tmp.replace(path)
    return path.stat().st_size


def append_stock(ticker: str, new: pd.DataFrame, chain_dir=None, keep: str = "new") -> int:
    """Append days to one stock's file. keep='new' lets the new rows replace
    existing rows for the same contract and day; keep='old' only fills days
    the file does not have yet."""
    p = paths(chain_dir=chain_dir)["chains"] / asset_name(ticker)
    new = normalize_chain(new, "unknown")
    if p.exists():
        old = pd.read_parquet(p)
        if keep == "old":
            new = new[~new["date"].isin(set(pd.to_datetime(old["date"]).dt.date))]
            if new.empty:
                return 0
        if keep == "new":            # a re-collected day replaces that whole day
            old = old[~pd.to_datetime(old["date"]).dt.date.isin(set(new["date"]))]
        both = pd.concat([old, new], ignore_index=True)
        both = normalize_chain(both, "unknown")
        both = both.drop_duplicates(subset=KEY, keep="last")
    else:
        both = new
    _write(both, p)
    return len(new)


def read_chain(ticker: str, start=None, end=None, chain_dir=None, download: bool = True) -> pd.DataFrame:
    """The full contract-level chain for one stock (all days). Downloads the
    stock's file from the GitHub Release if there is no local copy from today."""
    p = paths(chain_dir=chain_dir)["chains"] / asset_name(ticker)
    fresh = p.exists() and dt.date.fromtimestamp(p.stat().st_mtime) == dt.date.today()
    if download and not fresh:
        try:
            import io
            import zipfile

            import requests
            r = requests.get(release_url(ticker), timeout=300)
            if r.status_code == 200:
                p.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:   # unpack the whole bundle: same-letter stocks
                    for n in z.namelist():
                        if n.endswith(".parquet"):
                            (p.parent / n).write_bytes(z.read(n))
        except Exception:
            pass
    if not p.exists():
        return pd.DataFrame(columns=CHAIN_COLS)
    filt = []
    if start:
        filt.append(("date", ">=", pd.Timestamp(start).date()))
    if end:
        filt.append(("date", "<=", pd.Timestamp(end).date()))
    return pq.read_table(p, filters=filt or None).to_pandas()


# ---------------------------------------------------------------- summary ---

def summarize(chain: pd.DataFrame, features: pd.DataFrame | None = None, gex_scale: float = 1.0,
              compute_gex: bool = False, rate: float = 0.045) -> pd.DataFrame:
    """One row per (date, ticker). GEX levels come from Watchtower's features
    when given (older days); otherwise they are computed here with Watchtower's
    method (gex.daily_levels), so both kinds of day form one series."""
    rows = []
    feat = None
    if features is not None and len(features):
        feat = features.set_index(["date", "ticker"])
    for (d, t), g in chain.groupby(["date", "ticker"], sort=True):
        calls, puts = g[g["is_call"]], g[~g["is_call"]]
        coi, poi = float(calls["oi"].sum()), float(puts["oi"].sum())
        has_vol = g["volume"].notna().any()
        cv = float(calls["volume"].sum()) if has_vol else np.nan
        pv = float(puts["volume"].sum()) if has_vol else np.nan
        ivok = g[(g["iv"] > IV_JUNK_LOW) & (g["iv"] < IV_JUNK_HIGH)]
        ivok = ivok[ivok["volume"] > 0] if has_vol else ivok[ivok["oi"] > 0]
        spot = float(g["spot"].dropna().iloc[0]) if g["spot"].notna().any() else np.nan
        r = {"date": d, "ticker": t, "spot": spot, "call_oi": coi, "put_oi": poi,
             "pc_oi": poi / coi if coi > 0 else np.nan, "call_volume": cv, "put_volume": pv,
             "pc_volume": pv / cv if (has_vol and cv > 0) else np.nan,
             "avg_iv": float(ivok["iv"].mean()) if len(ivok) else np.nan, "n_contracts": len(g),
             "source": str(g["source"].iloc[-1])}
        if feat is not None and (d, t) in feat.index:
            f = feat.loc[(d, t)]
            r.update({"spot": f.get("spot", spot), "gex_total": f.get("gex_total"), "gex_flip": f.get("flip_level"),
                      "call_wall_price": f.get("call_wall"), "call_wall_gex": f.get("call_wall_gex"),
                      "put_wall_price": f.get("put_wall"), "put_wall_gex": f.get("put_wall_gex"),
                      "ivx_30d": f.get("ivx_30d"), "gex_source": "watchtower"})
            # walls at strikes (README "GEX levels"); Watchtower's flip and total GEX are kept
            r.update(G.strike_walls(G._filtered(g), r["spot"], rate)); r["wall_method"] = "strike"
        elif compute_gex and g["iv"].notna().any() and np.isfinite(spot):
            r.update(G.daily_levels(g, spot, rate)); r["gex_source"] = "computed"
        rows.append(r)
    out = pd.DataFrame(rows)
    for c in SUMMARY_COLS:
        if c not in out:
            out[c] = np.nan
    return out[SUMMARY_COLS]


def upsert_summary(new: pd.DataFrame, store=None, prefer_existing_cols: tuple = ()):
    """Add or replace (date, ticker) rows; one file per month so only the current
    month's small file changes on a normal day."""
    P = paths(store); P["summary"].mkdir(parents=True, exist_ok=True)
    new = new.copy()
    for c in SUMMARY_COLS:
        if c not in new:
            new[c] = np.nan
    new = new[SUMMARY_COLS]
    new["date"] = pd.to_datetime(new["date"]).dt.date
    new["month"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m")
    for mon, part in new.groupby("month"):
        f = P["summary"] / f"{mon}.parquet"
        part = part.drop(columns="month")
        if f.exists():
            old = pd.read_parquet(f)
            for c in SUMMARY_COLS:
                if c not in old:
                    old[c] = np.nan
            old["date"] = pd.to_datetime(old["date"]).dt.date
            merged = old.merge(part, on=["date", "ticker"], how="outer", suffixes=("_old", ""))
            for c in SUMMARY_COLS:
                if c in ("date", "ticker") or c + "_old" not in merged:
                    continue
                if c in prefer_existing_cols:
                    merged[c] = merged[c + "_old"].combine_first(merged[c])
                else:
                    merged[c] = merged[c].combine_first(merged[c + "_old"])
                merged = merged.drop(columns=c + "_old")
            part = merged[SUMMARY_COLS]
        part = part.sort_values(["date", "ticker"]).reset_index(drop=True)
        part.to_parquet(f, index=False, compression="zstd")


def read_summary(tickers=None, store=None) -> pd.DataFrame:
    P = paths(store)
    files = sorted(P["summary"].glob("*.parquet")) if P["summary"].exists() else []
    if not files:
        return pd.DataFrame(columns=SUMMARY_COLS)
    filt = [("ticker", "in", [t.upper() for t in tickers])] if tickers else None
    df = pd.concat([pq.read_table(f, filters=filt).to_pandas() for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def market_put_call(store=None) -> pd.DataFrame:
    """Market-wide put/call ratios from all single stocks (index ETFs excluded):
    a home-made stand-in for Cboe's equity-only ratio."""
    s = read_summary(store=store)
    if s.empty:
        return pd.DataFrame(columns=["pc_oi", "pc_volume"])
    s = s[~s["ticker"].isin(INDEX_ETFS)]
    g = s.groupby("date")[["put_oi", "call_oi", "put_volume", "call_volume"]].sum(min_count=1)
    return pd.DataFrame({"pc_oi": g["put_oi"] / g["call_oi"], "pc_volume": g["put_volume"] / g["call_volume"]})


def load_state(store=None) -> dict:
    p = paths(store)["state"]
    return json.loads(p.read_text()) if p.exists() else {"labels": {}, "last_sync": None}


def save_state(st: dict, store=None):
    p = paths(store)["state"]; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1, sort_keys=True, default=str))


def status(store=None) -> dict:
    s = read_summary(store=store)
    st = load_state(store)
    return {"summary_days": int(s["date"].nunique()) if len(s) else 0,
            "summary_first": s["date"].min().date() if len(s) else None,
            "summary_last": s["date"].max().date() if len(s) else None,
            "tickers": int(s["ticker"].nunique()) if len(s) else 0,
            "volume_days": int(s.loc[s["pc_volume"].notna(), "date"].nunique()) if len(s) else 0,
            "last_sync": st.get("last_sync"), "last_collect": _last_collect(store), "release": f"https://github.com/{REPO}/releases/tag/{RELEASE_TAG}"}


def _last_collect(store=None):
    p = paths(store)["root"] / "collect_state.json"
    if not p.exists():
        return None
    st = json.loads(p.read_text())
    if not st:
        return None
    day = max(st)
    v = st[day]
    return f"session {day}: {v['collected']} of {v['stocks']} stocks (last run {v['last_run'].replace('T', ' ')})"


def universe(store=None) -> list[str]:
    p = paths(store)["universe"]
    if p.exists():
        return [x.strip().upper() for x in p.read_text().split() if x.strip()]
    return []
