"""One-time import of the existing history into options_data/.

    python -m atrade.import_history --chain chain_history.parquet --oi oi_history.parquet --gex gex_history.csv

What it does
  1. chain_history.parquet  -> full chain parts (source "chain_history")
  2. oi_history.parquet     -> added to the chain for the dates BEFORE the chain
                               starts (source "oi_history"; no spot, IV or expiry)
  3. gex_history.csv        -> GEX flip and walls in the daily summary
  4. Builds the daily summary (put/call ratios from open interest, average IV,
     spot, GEX) and universe.txt (the tickers the daily update will collect).
  5. Calibrates this tool's GEX formula against gex_history.csv, so new daily
     values continue the old series, and prints a comparison report.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from . import gex as G
from . import options_store as OS


def read_gex_csv(path) -> pd.DataFrame:
    g = pd.read_csv(path)
    g.columns = [c.strip() for c in g.columns]
    ren = {"Date": "date", "Ticker": "ticker", "Spot": "spot", "Flip": "gex_flip",
           "Call_Wall_Price": "call_wall_price", "Call_Wall_GEX": "call_wall_gex",
           "Put_Wall_Price": "put_wall_price", "Put_Wall_GEX": "put_wall_gex"}
    g = g.rename(columns=ren)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["ticker"] = g["ticker"].astype(str).str.upper()
    g = g.drop_duplicates(["date", "ticker"], keep="last")
    g["gex_source"] = "gex_history"
    return g[["date", "ticker", "spot", "gex_flip", "call_wall_price", "call_wall_gex",
              "put_wall_price", "put_wall_gex", "gex_source"]]


def calibrate(chain: pd.DataFrame, gx: pd.DataFrame, n_days: int = 5, max_tickers: int = 40) -> dict:
    """Recompute GEX from the chain with scale 1 on a sample of overlapping
    (date, ticker) pairs and compare with gex_history.csv."""
    both = chain[["date", "ticker"]].drop_duplicates().merge(gx[["date", "ticker"]], on=["date", "ticker"])
    if both.empty:
        return {"pairs": 0}
    days = sorted(both["date"].unique())[-n_days:]
    sample = both[both["date"].isin(days)]
    tick = list(dict.fromkeys(["SPY"] + sorted(sample["ticker"].unique())))[:max_tickers]
    sample = sample[sample["ticker"].isin(tick)]
    rows = []
    for d, t in sample.itertuples(index=False):
        c = chain[(chain["date"] == d) & (chain["ticker"] == t)]
        spot = float(c["spot"].iloc[0])
        mine = G.levels(c, spot, scale=1.0)
        theirs = gx[(gx["date"] == d) & (gx["ticker"] == t)].iloc[0]
        rows.append({"date": d, "ticker": t, "spot": spot,
                     "cw_mine": mine["call_wall_price"], "cw_theirs": theirs["call_wall_price"],
                     "pw_mine": mine["put_wall_price"], "pw_theirs": theirs["put_wall_price"],
                     "flip_mine": mine["gex_flip"], "flip_theirs": theirs["gex_flip"],
                     "ratio_cw": theirs["call_wall_gex"] / mine["call_wall_gex"] if mine["call_wall_gex"] else np.nan,
                     "ratio_pw": theirs["put_wall_gex"] / mine["put_wall_gex"] if mine["put_wall_gex"] else np.nan})
    r = pd.DataFrame(rows)
    ratios = pd.concat([r["ratio_cw"], r["ratio_pw"]]).replace([np.inf, -np.inf], np.nan).dropna()
    ratios = ratios[ratios > 0]
    return {
        "pairs": len(r),
        "scale": float(np.median(ratios)) if len(ratios) else 1.0,
        "scale_spread": float(ratios.quantile(0.75) / ratios.quantile(0.25)) if len(ratios) > 3 else np.nan,
        "call_wall_match": float((np.isclose(r["cw_mine"], r["cw_theirs"])).mean()),
        "put_wall_match": float((np.isclose(r["pw_mine"], r["pw_theirs"])).mean()),
        "flip_median_gap_pct": float(((r["flip_mine"] - r["flip_theirs"]).abs() / r["spot"]).median()),
        "detail": r,
    }


def main(chain_path, oi_path, gex_path, store=None):
    t0 = dt.datetime.now()
    P = OS.paths(store)
    if list(P["chain"].glob("part_*.parquet")):
        raise SystemExit(f"{P['chain']} already has history parts. Delete options_data/ first to re-import.")
    print("Reading chain_history ...")
    chain = OS.normalize_chain(pd.read_parquet(chain_path), "chain_history")
    first_chain = chain["date"].min()
    print(f"  {len(chain):,} rows, {chain['ticker'].nunique()} tickers, {first_chain} to {chain['date'].max()}")

    oi_rows = pd.DataFrame()
    if oi_path:
        print("Reading oi_history ...")
        oi = pd.read_parquet(oi_path)
        oi["date"] = pd.to_datetime(oi["date"]).dt.date
        oi_rows = oi[oi["date"] < first_chain].copy()
        print(f"  {len(oi):,} rows; {len(oi_rows):,} rows from before {first_chain} are added to the chain store")
        oi_rows = OS.normalize_chain(oi_rows.assign(source="oi_history"), "oi_history")

    gx = read_gex_csv(gex_path) if gex_path else pd.DataFrame()
    if len(gx):
        print(f"Read gex_history: {len(gx):,} rows, {gx['date'].min()} to {gx['date'].max()}")

    print("Writing chain parts (each under 95 MB) ...")
    full = pd.concat([oi_rows, chain], ignore_index=True) if len(oi_rows) else chain
    parts = OS.write_history_parts(full, store)
    for p in parts:
        print(f"  {p['file']}: {p['first']} to {p['last']}, {p['rows']:,} rows, {p['bytes'] / 2**20:.1f} MB")

    cal = {"pairs": 0, "scale": 1.0}
    if len(gx):
        print("Calibrating GEX against gex_history.csv ...")
        cal = calibrate(chain, gx)
        if cal["pairs"]:
            print(f"  {cal['pairs']} stock-days compared. Scale factor {cal['scale']:.4g} "
                  f"(75th/25th percentile ratio {cal['scale_spread']:.3g}).")
            print(f"  Call wall same strike: {cal['call_wall_match']:.0%}; put wall same strike: {cal['put_wall_match']:.0%}; "
                  f"median flip gap {cal['flip_median_gap_pct']:.2%} of spot.")
            cal["detail"].to_csv(P["root"] / "gex_calibration.csv", index=False)

    print("Building the daily summary ...")
    summ = OS.summarize(chain, compute_gex=False)
    if len(oi_rows):
        oi_sum = OS.summarize(oi_rows, compute_gex=False)
        summ = pd.concat([oi_sum, summ], ignore_index=True)
    if len(gx):
        summ = summ.drop(columns=["gex_flip", "call_wall_price", "call_wall_gex", "put_wall_price", "put_wall_gex",
                                  "gex_source", "gex_total"])
        summ = summ.merge(gx.rename(columns={"spot": "spot_gex"}), on=["date", "ticker"], how="outer")
        summ["spot"] = summ["spot"].combine_first(summ["spot_gex"])
        summ = summ.drop(columns="spot_gex")
    # chain days without gex_history rows: compute GEX here
    need = summ[summ["gex_source"].isna() & summ["date"].isin(set(chain["date"]))][["date", "ticker"]] \
        if "gex_source" in summ else summ[["date", "ticker"]].iloc[0:0]
    if len(need):
        print(f"  computing GEX for {len(need):,} stock-days not in gex_history ...")
        sub = chain.merge(need, on=["date", "ticker"])
        comp = OS.summarize(sub, gex_scale=cal.get("scale", 1.0))[
            ["date", "ticker", "gex_total", "gex_flip", "call_wall_price", "call_wall_gex", "put_wall_price",
             "put_wall_gex", "gex_source"]]
        summ = summ.set_index(["date", "ticker"]); comp = comp.set_index(["date", "ticker"])
        summ.update(comp); summ = summ.reset_index()
    summ["source"] = summ["source"].fillna("gex_history")
    for c in OS.SUMMARY_COLS:
        if c not in summ:
            summ[c] = np.nan
    OS.upsert_summary(summ[OS.SUMMARY_COLS], store)

    tickers = sorted(set(chain["ticker"]) | (set(gx["ticker"]) if len(gx) else set()))
    P["universe"].write_text("\n".join(tickers) + "\n")
    m = OS.load_manifest(store)
    m.update({"gex_scale": cal.get("scale", 1.0), "imported": dt.datetime.now().isoformat(timespec="minutes"),
              "last_update": str(chain["date"].max())})
    OS.save_manifest(m, store)
    st = OS.status(store)
    print(f"Done in {(dt.datetime.now() - t0).seconds}s: {st['summary_days']} days x {st['tickers']} tickers in the summary "
          f"({st['summary_first']} to {st['summary_last']}); {len(st['parts'])} chain part(s); universe of {len(tickers)} tickers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", required=True); ap.add_argument("--oi"); ap.add_argument("--gex")
    a = ap.parse_args()
    main(a.chain, a.oi, a.gex)
