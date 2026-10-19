"""The consolidated options history kept in the repo (folder options_data/).

    options_data/
      summary/2026-10.parquet       one row per stock per day (quick use), one file per month
      chain/part_001.parquet ...    full contract-level chain, each part < 95 MB, never rewritten
      chain/pending/2026-10-19.parquet   recent days not yet merged into a part
      universe.txt                  tickers the daily update collects
      manifest.json                 what is in each part, last update, GEX calibration

Why daily 'pending' files: git keeps every version of every file. Rewriting a
large part every day would grow the repo by the part's size each day. Instead
each day is written once to its own small file; when the pending files add
up to about 95 MB they are merged into the next numbered part, which is
never changed again. read_chain() joins all parts and pending files back
into the full chain.

Chain columns: date, ticker, spot, expiry_str, strike, is_call, iv, oi, T, volume, source
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
from .config import ROOT

STORE = ROOT / "options_data"
MAX_PART_BYTES = 95 * 1024 * 1024
INDEX_ETFS = {"SPY", "QQQ", "IWM", "DIA", "SPX", "^SPX", "XSP", "OEX"}

CHAIN_SCHEMA = pa.schema([
    ("date", pa.date32()), ("ticker", pa.string()), ("spot", pa.float64()), ("expiry_str", pa.string()),
    ("strike", pa.float64()), ("is_call", pa.bool_()), ("iv", pa.float64()), ("oi", pa.float64()),
    ("T", pa.float64()), ("volume", pa.float64()), ("source", pa.string()),
])
SUMMARY_COLS = ["date", "ticker", "spot", "call_oi", "put_oi", "pc_oi", "call_volume", "put_volume", "pc_volume",
                "avg_iv", "n_contracts", "gex_total", "gex_flip", "call_wall_price", "call_wall_gex",
                "put_wall_price", "put_wall_gex", "gex_source", "source"]


def paths(store: Path | None = None):
    s = Path(store) if store else STORE
    return {"root": s, "summary": s / "summary", "chain": s / "chain", "pending": s / "chain" / "pending",
            "manifest": s / "manifest.json", "universe": s / "universe.txt"}


# --------------------------------------------------------------- helpers ---

def normalize_chain(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df.attrs = {}
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "expiry_str" not in df:
        df["expiry_str"] = None
    df["expiry_str"] = df["expiry_str"].astype("string").where(df["expiry_str"].notna(), None)
    for col in ("spot", "iv", "oi", "T", "volume"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype(float)
    df["is_call"] = df["is_call"].astype(bool)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if "source" not in df:
        df["source"] = source
    df = df[[f.name for f in CHAIN_SCHEMA]]
    return df.sort_values(["date", "ticker", "expiry_str", "strike", "is_call"], na_position="first").reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(df, schema=CHAIN_SCHEMA, preserve_index=False)
    pq.write_table(tbl, path, compression="zstd", compression_level=9, row_group_size=250_000)
    return path.stat().st_size


def _write_sized(df: pd.DataFrame, chain_dir: Path, first_no: int, max_bytes: int) -> list[dict]:
    """Write df as one or more parts, splitting by date until each is < max_bytes."""
    dates = sorted(df["date"].unique())
    out = []

    def go(ds, no):
        tmp = chain_dir / f"part_{no:03d}.parquet"
        size = _write(df[df["date"].isin(ds)], tmp)
        if size <= max_bytes or len(ds) == 1:
            out.append({"file": tmp.name, "first": str(ds[0]), "last": str(ds[-1]), "bytes": size,
                        "rows": int(df["date"].isin(ds).sum())})
            return no + 1
        tmp.unlink()
        half = len(ds) // 2
        no = go(ds[:half], no)
        return go(ds[half:], no)

    go(dates, first_no)
    return out


def load_manifest(store=None) -> dict:
    p = paths(store)["manifest"]
    if p.exists():
        return json.loads(p.read_text())
    return {"parts": [], "gex_scale": G.GEX_SCALE, "last_update": None}


def save_manifest(m: dict, store=None):
    p = paths(store)["manifest"]; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, default=str))


# -------------------------------------------------------------- chain I/O ---

def add_chain_day(day_df: pd.DataFrame, store=None, max_bytes: int = MAX_PART_BYTES) -> dict:
    """Store one day of the chain as a pending file, then merge pending files into
    a new part if they add up to max_bytes."""
    P = paths(store)
    day_df = normalize_chain(day_df, "yahoo")
    d = day_df["date"].iloc[0]
    _write(day_df, P["pending"] / f"{d}.parquet")
    return compact(store, max_bytes)


def compact(store=None, max_bytes: int = MAX_PART_BYTES, force: bool = False) -> dict:
    P = paths(store); m = load_manifest(store)
    pend = sorted(P["pending"].glob("*.parquet")) if P["pending"].exists() else []
    total = sum(f.stat().st_size for f in pend)
    if not pend or (total < max_bytes and not force):
        return {"compacted": False, "pending_files": len(pend), "pending_bytes": total}
    df = pd.concat([pd.read_parquet(f) for f in pend], ignore_index=True)
    df = normalize_chain(df, "yahoo")
    no = len(m["parts"]) + 1
    new = _write_sized(df, P["chain"], no, max_bytes)
    m["parts"] += new
    save_manifest(m, store)
    for f in pend:
        f.unlink()
    return {"compacted": True, "new_parts": new}


def write_history_parts(df: pd.DataFrame, store=None, max_bytes: int = MAX_PART_BYTES) -> list[dict]:
    """One-time import: write a big history as sealed parts."""
    P = paths(store); m = load_manifest(store)
    new = _write_sized(normalize_chain(df, "import"), P["chain"], len(m["parts"]) + 1, max_bytes)
    m["parts"] += new
    save_manifest(m, store)
    return new


def chain_files(store=None) -> list[Path]:
    P = paths(store)
    parts = sorted(P["chain"].glob("part_*.parquet")) if P["chain"].exists() else []
    pend = sorted(P["pending"].glob("*.parquet")) if P["pending"].exists() else []
    return parts + pend


def read_chain(tickers=None, start=None, end=None, store=None, columns=None) -> pd.DataFrame:
    """Rebuild the full contract-level chain from every part and pending file."""
    filters = []
    if tickers:
        filters.append(("ticker", "in", [t.upper() for t in tickers]))
    if start:
        filters.append(("date", ">=", pd.Timestamp(start).date()))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end).date()))
    frames = []
    for f in chain_files(store):
        t = pq.read_table(f, columns=columns, filters=filters or None)
        if t.num_rows:
            frames.append(t.to_pandas())
    if not frames:
        return pd.DataFrame(columns=columns or [x.name for x in CHAIN_SCHEMA])
    df = pd.concat(frames, ignore_index=True)
    # a day re-collected later replaces the earlier copy
    if "source" in df and "date" in df:
        df = df.drop_duplicates(subset=[c for c in ("date", "ticker", "expiry_str", "strike", "is_call") if c in df],
                                keep="last")
    return df


# ---------------------------------------------------------------- summary ---

def summarize(chain: pd.DataFrame, gex_scale: float | None = None, compute_gex: bool = True) -> pd.DataFrame:
    """One row per (date, ticker) from contract-level rows."""
    rows = []
    for (d, t), g in chain.groupby(["date", "ticker"], sort=True):
        spot = float(g["spot"].dropna().iloc[0]) if g["spot"].notna().any() else np.nan
        calls, puts = g[g["is_call"]], g[~g["is_call"]]
        coi, poi = float(calls["oi"].sum()), float(puts["oi"].sum())
        has_vol = g["volume"].notna().any()
        cv = float(calls["volume"].sum()) if has_vol else np.nan
        pv = float(puts["volume"].sum()) if has_vol else np.nan
        ivok = g[(g["iv"] > G.IV_MIN) & (g["iv"] < G.IV_MAX)]
        ivok = ivok[ivok["volume"] > 0] if has_vol else ivok[ivok["oi"] > 0]
        r = {"date": d, "ticker": t, "spot": spot, "call_oi": coi, "put_oi": poi,
             "pc_oi": poi / coi if coi > 0 else np.nan, "call_volume": cv, "put_volume": pv,
             "pc_volume": pv / cv if (has_vol and cv > 0) else np.nan,
             "avg_iv": float(ivok["iv"].mean()) if len(ivok) else np.nan, "n_contracts": len(g),
             "source": str(g["source"].iloc[-1])}
        if compute_gex and g["iv"].notna().any():
            r.update(G.levels(g, spot, gex_scale)); r["gex_source"] = "computed"
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
    new = new.copy(); new["date"] = pd.to_datetime(new["date"]).dt.date
    new["month"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m")
    for mon, part in new.groupby("month"):
        f = P["summary"] / f"{mon}.parquet"
        part = part.drop(columns="month")
        if f.exists():
            old = pd.read_parquet(f)
            old["date"] = pd.to_datetime(old["date"]).dt.date
            merged = old.merge(part, on=["date", "ticker"], how="outer", suffixes=("_old", ""))
            for c in SUMMARY_COLS:
                if c in ("date", "ticker"):
                    continue
                if c + "_old" in merged:
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


def status(store=None) -> dict:
    P = paths(store); m = load_manifest(store)
    s = read_summary(store=store)
    pend = sorted(P["pending"].glob("*.parquet")) if P["pending"].exists() else []
    return {"parts": m["parts"], "pending_files": len(pend),
            "pending_mb": sum(f.stat().st_size for f in pend) / 2 ** 20,
            "summary_days": int(s["date"].nunique()) if len(s) else 0,
            "summary_first": s["date"].min().date() if len(s) else None,
            "summary_last": s["date"].max().date() if len(s) else None,
            "tickers": int(s["ticker"].nunique()) if len(s) else 0,
            "volume_days": int(s.loc[s["pc_volume"].notna(), "date"].nunique()) if len(s) else 0,
            "last_update": m.get("last_update"), "gex_scale": m.get("gex_scale")}


def universe(store=None) -> list[str]:
    p = paths(store)["universe"]
    if p.exists():
        return [x.strip().upper() for x in p.read_text().split() if x.strip()]
    return []
