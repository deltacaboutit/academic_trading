"""Copy new days from the Watchtower repo into this tool's options store.

    python -m atrade.watchtower --work /tmp/watchtower            # new or changed days only
    python -m atrade.watchtower --work /tmp/watchtower --labels 2026-10-16

Watchtower (github.com/GittHubbard/Watchtower) collects the Yahoo Finance
chain of every stock each evening and saves
    data/chains/ticker=XXX/date=LABEL.parquet   (one file per stock per day)
    data/features/date=LABEL.parquet            (spot, GEX flip and walls, ...)

Three fixes are applied on the way in:
  1. Real trading day. LABEL is the UTC date when Watchtower's run STARTED, so a
     run that starts after 8 pm New York time (or a weekend re-run) carries the
     wrong date. The real session is the New York date of the latest option
     trade in that run (snapped back to an NYSE trading day). If two runs map
     to the same session, the MORE COMPLETE one (more contracts) is kept, so a
     broken partial run never overwrites a full one.
  2. Junk implied volatility is filled by interpolation between the nearest
     good strikes of the same expiry and type (options_store.clean_iv).
  3. Columns are renamed to this tool's chain layout and each stock's day is
     appended to that stock's single file (data/chains/<TICKER>.parquet).

The daily summary (put/call ratios, average IV, spot and Watchtower's GEX
levels) is added to options_data/summary/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import options_store as OS

WT_URL = "https://github.com/GittHubbard/Watchtower"
NY = ZoneInfo("America/New_York")
MIN_CONTRACTS = 50_000      # a full run has ~120-150k contracts; below this it is a broken partial run


def _git(work, *args, check=True):
    r = subprocess.run(["git", "-C", str(work), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def clone(work: Path):
    work = Path(work)
    if (work / ".git").exists():
        _git(work, "fetch", "--depth", "1", "--filter=blob:none", "origin", "HEAD")
        _git(work, "reset", "--soft", "FETCH_HEAD")
    else:
        work.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", WT_URL, str(work)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
    _git(work, "sparse-checkout", "init", "--no-cone")
    return _git(work, "rev-parse", "HEAD").strip()


def list_labels(work: Path) -> dict:
    """{label: blob id of that day's features file}."""
    out = {}
    for line in _git(work, "ls-tree", "-r", "HEAD", "data/features").splitlines():
        meta, path = line.split("\t")
        name = Path(path).name
        if name.startswith("date=") and name.endswith(".parquet"):
            out[name[5:15]] = meta.split()[2]
    return out


def checkout(work: Path, labels: list[str]):
    pats = []
    for L in labels:
        pats += [f"/data/chains/*/date={L}.parquet", f"/data/features/date={L}.parquet"]
    _git(work, "sparse-checkout", "set", "--no-cone", *pats)
    _git(work, "read-tree", "-mu", "HEAD")


def trading_day_on_or_before(d: dt.date) -> dt.date:
    try:
        import pandas_market_calendars as mcal
        sch = mcal.get_calendar("NYSE").schedule(start_date=d - dt.timedelta(days=10), end_date=d)
        return sch.index[-1].date()
    except Exception:
        while d.weekday() >= 5:
            d -= dt.timedelta(days=1)
        return d


def read_label(work: Path, label: str):
    """All chains + features of one Watchtower run, with the real session date."""
    feats = pd.read_parquet(work / "data" / "features" / f"date={label}.parquet")
    frames = []
    for f in sorted((work / "data" / "chains").glob(f"ticker=*/date={label}.parquet")):
        c = pd.read_parquet(f)
        c["ticker"] = f.parent.name.split("=", 1)[1].upper()
        frames.append(c)
    if not frames:
        return None, None, None
    ch = pd.concat(frames, ignore_index=True)
    last = pd.to_datetime(ch["lastTradeDate"], utc=True, errors="coerce").max()
    session = trading_day_on_or_before(last.tz_convert(NY).date()) if pd.notna(last) else \
        trading_day_on_or_before(dt.date.fromisoformat(label))
    ch["date"] = session
    spot = feats.set_index(feats["ticker"].str.upper())["spot"]
    ch["spot"] = ch["ticker"].map(spot)
    ch["source"] = "watchtower"
    ch = OS.clean_iv(OS.normalize_chain(ch, "watchtower"))
    feats = feats.assign(date=session, ticker=feats["ticker"].str.upper())
    return session, ch, feats


def sync(work: Path, store=None, chain_dir=None, labels: list[str] | None = None, batch: int = 8, log=print) -> dict:
    head = clone(work)
    st = OS.load_state(store)
    have = list_labels(work)
    if labels:
        todo = [L for L in labels if L in have]
    else:
        todo = sorted(L for L, sha in have.items() if st["labels"].get(L, {}).get("sha") != sha)
    log(f"Watchtower {head[:8]}: {len(have)} days available, {len(todo)} new or changed.")
    changed: set[str] = set()
    sessions = {}
    summ_all = OS.read_summary(store=store)
    own = set(summ_all.loc[summ_all["source"] == "yahoo", "date"].dt.date) if len(summ_all) else set()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        checkout(work, chunk)
        per_stock = defaultdict(list)
        summ = []
        for L in chunk:                              # label order: later labels win
            session, ch, feats = read_label(work, L)
            if session is None:
                log(f"  {L}: no chain files, skipped")
                continue
            st["labels"][L] = {"sha": have[L], "session": str(session), "contracts": len(ch)}
            full = [v["contracts"] for v in st.setdefault("sessions", {}).values()]
            floor = 0.5 * float(np.median(full)) if len(full) >= 3 else MIN_CONTRACTS
            if len(ch) < floor:
                st["labels"][L]["partial"] = True
                log(f"  {L} -> trading day {session}: only {len(ch):,} contracts (below the {floor:,.0f} minimum) - "
                    f"PARTIAL RUN, skipped; this day stays missing until you import it")
                continue
            if session in own:
                log(f"  {L} -> trading day {session}: SKIPPED, this repo's own collector already has that day")
                continue
            prev = st["sessions"].get(str(session))
            if prev and prev["label"] != L and prev["contracts"] > len(ch):
                log(f"  {L} -> trading day {session}: {len(ch):,} contracts - SKIPPED, run {prev['label']} "
                    f"has more ({prev['contracts']:,})")
                continue
            # a better run for a session already queued in this batch replaces it
            for t in list(per_stock):
                per_stock[t] = [g for g in per_stock[t] if g["date"].iloc[0] != session]
            summ = [x for x in summ if not (pd.to_datetime(x["date"]).dt.date == session).all()]
            for t, g in ch.groupby("ticker"):
                per_stock[t].append(g)
            summ.append(OS.summarize(ch, feats))
            st["sessions"][str(session)] = {"label": L, "contracts": len(ch)}
            sessions[L] = session
            log(f"  {L} -> trading day {session}: {ch['ticker'].nunique()} stocks, {len(ch):,} contracts, "
                f"{int(ch['iv_filled'].sum()):,} IVs filled")
        for t, parts in per_stock.items():
            OS.append_stock(t, pd.concat(parts, ignore_index=True), chain_dir=chain_dir, keep="new")
            changed.add(t)
        if summ:
            OS.upsert_summary(pd.concat(summ, ignore_index=True), store)
    P = OS.paths(store)
    uni = sorted(set(OS.universe(store)) | changed)
    if uni:
        P["universe"].parent.mkdir(parents=True, exist_ok=True)
        P["universe"].write_text("\n".join(uni) + "\n")
    st["last_sync"] = dt.datetime.now(NY).isoformat(timespec="minutes")
    st["watchtower_commit"] = head
    OS.save_state(st, store)
    return {"labels": len(todo), "stocks_changed": len(changed), "sessions": sessions}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/watchtower")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args(argv)
    r = sync(Path(a.work), labels=a.labels, batch=a.batch)
    print(f"Done: {r['labels']} Watchtower days processed, {r['stocks_changed']} stock files updated.")


if __name__ == "__main__":
    sys.exit(main())
