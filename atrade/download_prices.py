"""Download daily price history for every stock the tool follows, for testing and validation.

    python -m atrade.download_prices                  # all tickers, 25 years
    python -m atrade.download_prices --years 30
    python -m atrade.download_prices --tickers AAPL MSFT SPY
    python -m atrade.download_prices --missing-only   # only tickers not downloaded yet / that failed
    python -m atrade.download_prices --info           # company info only -> price_data/info.csv

Tickers: options_data/universe.txt (the S&P 500 stocks plus SPY, QQQ and IWM
that Watchtower collects), plus the VIX index (^VIX).

Output: price_data/<TICKER>.parquet - one file per ticker, one row per trading day:
    date        trading day
    Open, High, Low, Close   adjusted for splits AND dividends (what the tool analyses)
    Volume      shares traded (adjusted for splits)
    Close_raw   the actual closing price on that day, adjusted for splits only - this is
                the price the option strikes in the options data refer to
    Dividends   cash dividend paid that day (0 on most days)
    Splits      split ratio on a split day (0 on other days)
and price_data/manifest.csv - one line per ticker: first and last date, number of days,
data checks, and any download error.

Files are written only after a ticker downloads completely, so a failed or interrupted run
never leaves a half-written file. Run it again with --missing-only to retry the failures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

OUT = ROOT / "price_data"
EXTRA = ["^VIX"]
COLS = ["Open", "High", "Low", "Close", "Volume", "Close_raw", "Dividends", "Splits"]


def file_name(ticker: str) -> str:
    return ticker.upper().replace("^", "_").replace("/", "-") + ".parquet"


def yahoo_symbol(ticker: str) -> str:
    return ticker.upper().replace(".", "-").replace("/", "-")


def fetch(ticker: str, years: int, tries: int = 4) -> pd.DataFrame:
    """One ticker from Yahoo Finance; retries with a growing pause (Yahoo rate-limits bursts)."""
    import yfinance as yf
    start = (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 7)).isoformat()
    err = None
    for k in range(tries):
        try:
            h = yf.Ticker(yahoo_symbol(ticker)).history(start=start, auto_adjust=False, actions=True)
            if h is None or h.empty:
                raise ValueError("Yahoo returned no rows")
            return to_table(h)
        except Exception as e:
            err = e
            time.sleep(4 * (k + 1))
    raise RuntimeError(str(err)[:200])


def to_table(h: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's unadjusted bars -> the layout above. Yahoo's 'Close' is split-adjusted only;
    'Adj Close' also takes dividends out, so the other prices are scaled by Adj Close / Close."""
    h = h.copy()
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    adj = h["Adj Close"] if "Adj Close" in h else h["Close"]
    f = (adj / h["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out = pd.DataFrame({
        "Open": h["Open"] * f, "High": h["High"] * f, "Low": h["Low"] * f, "Close": adj,
        "Volume": h["Volume"].astype(float), "Close_raw": h["Close"],
        "Dividends": h.get("Dividends", 0.0), "Splits": h.get("Stock Splits", 0.0),
    }, index=h.index)
    out.index.name = "date"
    return out[COLS].astype(float)


EXCHANGE_GROUP = {   # Yahoo exchange codes -> the two groups LMW report separately
    "NMS": "Nasdaq", "NGM": "Nasdaq", "NCM": "Nasdaq", "NAS": "Nasdaq",
    "NYQ": "NYSE/AMEX", "NYS": "NYSE/AMEX", "ASE": "NYSE/AMEX", "AMX": "NYSE/AMEX", "PCX": "NYSE/AMEX",
}


def fetch_info(ticker: str, tries: int = 3) -> dict:
    import yfinance as yf
    err = None
    for k in range(tries):
        try:
            i = yf.Ticker(yahoo_symbol(ticker)).info or {}
            ex = i.get("exchange") or ""
            return {"ticker": ticker, "name": i.get("shortName") or i.get("longName"), "exchange": ex,
                    "exchange_group": EXCHANGE_GROUP.get(ex, "Other"), "quote_type": i.get("quoteType"),
                    "sector": i.get("sector"), "industry": i.get("industry"),
                    "shares_outstanding": i.get("sharesOutstanding"), "market_cap": i.get("marketCap"),
                    "info_date": dt.date.today().isoformat()}
        except Exception as e:
            err = e
            time.sleep(3 * (k + 1))
    return {"ticker": ticker, "error": str(err)[:200]}


def download_info(tickers: list[str], out: Path, workers: int = 4, fetcher=None) -> pd.DataFrame:
    fetcher = fetcher or fetch_info
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for i, r in enumerate(pool.map(fetcher, tickers), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(tickers):
                print(f"  info {i}/{len(tickers)}")
    df = pd.DataFrame(rows).sort_values("ticker")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "info.csv", index=False)
    print(df["exchange_group"].value_counts(dropna=False).to_string() if "exchange_group" in df else "")
    return df


def checks(df: pd.DataFrame) -> dict:
    """Quick data checks (the full Framework 1 cleaning runs inside the tool)."""
    bad = ((df["High"] < df["Low"]) | (df["Close"] > df["High"] * 1.0001) | (df["Close"] < df["Low"] * 0.9999)
           | (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)).sum()
    r = df["Close"].pct_change().abs()
    gaps = (df.index.to_series().diff().dt.days > 5).sum()
    return {"bad_bars": int(bad), "moves_over_40pct": int((r > 0.40).sum()), "gaps_over_5_days": int(gaps),
            "zero_volume_days": int((df["Volume"] == 0).sum())}


def save(ticker: str, df: pd.DataFrame, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / file_name(ticker)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd")
    tmp.replace(p)                       # never leaves a half-written file
    return p


def load_universe() -> list[str]:
    p = ROOT / "options_data" / "universe.txt"
    tick = [t.strip().upper() for t in p.read_text().split()] if p.exists() else []
    if not tick:
        raise SystemExit("options_data/universe.txt not found - run 'git pull' first, or pass --tickers.")
    return tick + [t for t in EXTRA if t not in tick]


def main(argv=None, fetcher=None, out: Path = OUT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=25)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--missing-only", action="store_true", help="skip tickers that already have a file")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--info", action="store_true", help="download company info (exchange, shares) only")
    a = ap.parse_args(argv)
    if a.info:
        tickers = [t.upper() for t in a.tickers] if a.tickers else load_universe()
        print(f"Downloading company info for {len(tickers)} tickers ...")
        df = download_info(tickers, out, a.workers, fetcher=fetcher if fetcher and a.tickers else None)
        bad = int(df["error"].notna().sum()) if "error" in df else 0
        print(f"Saved price_data/info.csv ({len(df) - bad} ok, {bad} failed).")
        return 0
    fetcher = fetcher or (lambda t: fetch(t, a.years))
    tickers = [t.upper() for t in a.tickers] if a.tickers else load_universe()
    if a.missing_only:
        tickers = [t for t in tickers if not (out / file_name(t)).exists()]
    print(f"Downloading {len(tickers)} tickers ({a.years} years) into {out} ...")
    man_path = out / "manifest.csv"
    manifest = pd.read_csv(man_path).set_index("ticker").to_dict("index") if man_path.exists() else {}
    t0 = time.time(); ok = 0; failed = []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fetcher, t): t for t in tickers}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                df = f.result()
                save(t, df, out)
                manifest[t] = {"first": df.index[0].date(), "last": df.index[-1].date(), "days": len(df),
                               **checks(df), "error": ""}
                ok += 1
            except Exception as e:
                failed.append(t)
                manifest[t] = {**manifest.get(t, {}), "error": str(e)[:200]}
            if i % 25 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)} done, {len(failed)} failed, {time.time() - t0:.0f}s")
    m = pd.DataFrame.from_dict(manifest, orient="index").rename_axis("ticker").sort_index()
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(man_path)
    size = sum(p.stat().st_size for p in out.glob("*.parquet")) / 2 ** 20
    print(f"\nSaved {ok} tickers ({size:.0f} MB in price_data). Failed: {len(failed)}"
          + (f" -> {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}" if failed else ""))
    if failed:
        print("Run again with --missing-only to retry the failed tickers.")
    flagged = m[(m.get("bad_bars", 0) > 0) | (m.get("moves_over_40pct", 0) > 0)] if len(m) else m
    if len(flagged):
        print(f"{len(flagged)} tickers have bars worth a look (see manifest.csv: bad_bars, moves_over_40pct).")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
