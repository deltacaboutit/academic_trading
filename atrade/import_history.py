"""Import extra options history into the per-stock files and the daily summary.

    python -m atrade.import_history --chain chain_history.parquet --oi oi_history.parquet --gex gex_history.csv
    python -m atrade.import_history --watchtower-folder missing_days/

Every input only FILLS days a stock does not have yet; days already taken
from Watchtower are left alone (except GEX, which fills summary gaps).

  --chain   flat file: date, ticker, spot, expiry_str, strike, is_call, iv, oi, T (+ volume)
  --oi      flat file: date, ticker, strike, is_call, oi (no expiry) - only for days
            with no chain at all; gives the open-interest put/call ratio
  --gex     gex_history.csv: Date, Ticker, Spot, Flip, Call_Wall_Price, ... -> summary GEX
  --watchtower-folder   a folder laid out like Watchtower: chains/ticker=XXX/date=YYYY-MM-DD.parquet
            (+ features/date=YYYY-MM-DD.parquet); dates are mapped to the real trading day
            the same way as the nightly sync

Afterwards, upload the changed stock files:  python -m atrade.release upload
and commit options_data/ (summary).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import options_store as OS


def read_gex_csv(path) -> pd.DataFrame:
    g = pd.read_csv(path)
    g.columns = [c.strip() for c in g.columns]
    g = g.rename(columns={"Date": "date", "Ticker": "ticker", "Spot": "spot", "Flip": "gex_flip",
                          "Call_Wall_Price": "call_wall_price", "Call_Wall_GEX": "call_wall_gex",
                          "Put_Wall_Price": "put_wall_price", "Put_Wall_GEX": "put_wall_gex"})
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["ticker"] = g["ticker"].astype(str).str.upper()
    g = g.drop_duplicates(["date", "ticker"], keep="last")
    g["gex_source"] = "gex_history"
    return g[["date", "ticker", "spot", "gex_flip", "call_wall_price", "call_wall_gex",
              "put_wall_price", "put_wall_gex", "gex_source"]]


def _fill_chain(ch: pd.DataFrame, source: str, store, chain_dir, log) -> pd.DataFrame:
    ch = OS.clean_iv(OS.normalize_chain(ch.assign(source=source), source))
    for t, g in ch.groupby("ticker"):
        OS.append_stock(t, g, chain_dir=chain_dir, keep="old")
    log(f"  {source}: {ch['ticker'].nunique()} stocks, {ch['date'].nunique()} days offered")
    return ch


def import_watchtower_folder(folder: Path, store=None, chain_dir=None, log=print):
    from .watchtower import read_label
    folder = Path(folder)
    root = folder if (folder / "data").exists() else None
    if root is None:                      # accept chains/ + features/ directly under the folder
        root = folder.parent / (folder.name + "_wt")
        (root / "data").mkdir(parents=True, exist_ok=True)
        for sub in ("chains", "features"):
            src, dst = folder / sub, root / "data" / sub
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve(), target_is_directory=True)
    labels = sorted({p.name[5:15] for p in (root / "data" / "chains").glob("ticker=*/date=*.parquet")})
    summ = []
    for L in labels:
        if not (root / "data" / "features" / f"date={L}.parquet").exists():
            (root / "data" / "features").mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ticker": [], "spot": []}).to_parquet(root / "data" / "features" / f"date={L}.parquet")
        session, ch, feats = read_label(root, L)
        if session is None:
            continue
        for t, g in ch.groupby("ticker"):
            OS.append_stock(t, g, chain_dir=chain_dir, keep="old")
        summ.append(OS.summarize(ch, feats if len(feats) else None))
        log(f"  {L} -> trading day {session}: {ch['ticker'].nunique()} stocks, {len(ch):,} contracts")
    if summ:
        OS.upsert_summary(pd.concat(summ, ignore_index=True), store, prefer_existing_cols=tuple(OS.SUMMARY_COLS))


def inbox_args(folder: Path) -> list[str]:
    """Turn the files found in the import inbox into command-line arguments."""
    import zipfile
    folder = Path(folder)
    args = []
    for f in sorted(folder.iterdir()):
        n = f.name.lower()
        if n.startswith("chain_history") and n.endswith(".parquet"):
            args += ["--chain", str(f)]
        elif n.startswith("oi_history") and n.endswith(".parquet"):
            args += ["--oi", str(f)]
        elif n.startswith("gex_history") and n.endswith(".csv"):
            args += ["--gex", str(f)]
        elif n.endswith(".zip"):
            out = folder / (f.stem + "_unzipped")
            with zipfile.ZipFile(f) as z:
                z.extractall(out)
            hits = list(out.rglob("ticker=*"))
            if hits:
                root = hits[0].parent.parent       # folder holding chains/
                args += ["--watchtower-folder", str(root.parent if root.name == "data" else root)]
    return args


def main(argv=None, store=None, chain_dir=None, log=print):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain"); ap.add_argument("--oi"); ap.add_argument("--gex")
    ap.add_argument("--watchtower-folder")
    ap.add_argument("--inbox", help="folder of downloaded inbox files (file types detected by name)")
    a = ap.parse_args(argv)
    if a.inbox:
        extra = inbox_args(Path(a.inbox))
        if len(extra) > 1:
            if len(extra) > 2 and extra.count("--watchtower-folder") > 1:
                log("Note: only the first Watchtower zip is used per run.")
            a = ap.parse_args(extra)
        else:
            log("The inbox is empty - nothing to import."); return
    existing = OS.read_summary(store=store)
    have = set(zip(existing["date"].dt.date, existing["ticker"])) if len(existing) else set()

    if a.watchtower_folder:
        log("Importing the Watchtower-style folder ...")
        import_watchtower_folder(Path(a.watchtower_folder), store, chain_dir, log)

    gx = read_gex_csv(a.gex) if a.gex else None
    if a.chain:
        log("Importing the flat chain file ...")
        ch = pd.read_parquet(a.chain)
        ch["date"] = pd.to_datetime(ch["date"]).dt.date
        ch["ticker"] = ch["ticker"].astype(str).str.upper()
        new_keys = set(zip(ch["date"], ch["ticker"])) - have
        ch = ch[[k in new_keys for k in zip(ch["date"], ch["ticker"])]]
        if len(ch):
            ch = _fill_chain(ch, "chain_history", store, chain_dir, log)
            OS.upsert_summary(OS.summarize(ch, compute_gex=gx is None), store,
                              prefer_existing_cols=tuple(OS.SUMMARY_COLS))
            have |= set(zip(ch["date"], ch["ticker"]))
        log(f"  {len(new_keys):,} stock-days were new")
    if a.oi:
        log("Importing open-interest-only history ...")
        oi = pd.read_parquet(a.oi)
        oi["date"] = pd.to_datetime(oi["date"]).dt.date
        oi["ticker"] = oi["ticker"].astype(str).str.upper()
        oi = oi[[k not in have for k in zip(oi["date"], oi["ticker"])]]
        if len(oi):
            oi = _fill_chain(oi, "oi_history", store, chain_dir, log)
            OS.upsert_summary(OS.summarize(oi), store, prefer_existing_cols=tuple(OS.SUMMARY_COLS))
        log(f"  {oi[['date', 'ticker']].drop_duplicates().shape[0]:,} stock-days were new")
    if gx is not None:
        log("Adding GEX history to the summary (only where it is missing) ...")
        g = gx.copy()
        for c in OS.SUMMARY_COLS:
            if c not in g:
                g[c] = np.nan
        g["source"] = "gex_history"
        OS.upsert_summary(g[OS.SUMMARY_COLS], store, prefer_existing_cols=tuple(OS.SUMMARY_COLS))
    P = OS.paths(store)
    s = OS.read_summary(store=store)
    uni = sorted(set(OS.universe(store)) | set(s["ticker"]))
    P["universe"].parent.mkdir(parents=True, exist_ok=True)
    P["universe"].write_text("\n".join(uni) + "\n")
    st = OS.status(store)
    log(f"Summary now covers {st['summary_days']} days ({st['summary_first']} to {st['summary_last']}) for "
        f"{st['tickers']} tickers. Next: python -m atrade.release upload   then commit options_data/.")


if __name__ == "__main__":
    main()
