"""Daily options update (run by GitHub Actions at 4:30 pm New York time).

    python -m atrade.update_options            # normal run, with safety checks
    python -m atrade.update_options --force    # run even if outside the time window
    python -m atrade.update_options --tickers SPY AAPL

Collects every expiry of every ticker in options_data/universe.txt from Yahoo
Finance: strike, call/put, implied volatility, open interest and the day's
volume. Then it stores the day in the chain, adds the daily summary rows
(with real volume put/call ratios and GEX levels) and logs the run.

Notes on timing: at 4:30 pm the day's option volume is complete. Open
interest on Yahoo is the figure published overnight by the OCC, i.e. as of the
PREVIOUS trading day's close.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import options_store as OS

NY = ZoneInfo("America/New_York")
EARLIEST = dt.time(16, 15)


def is_trading_day(d: dt.date) -> bool:
    try:
        import pandas_market_calendars as mcal
        return not mcal.get_calendar("NYSE").schedule(start_date=d, end_date=d).empty
    except Exception:
        return d.weekday() < 5


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-").replace("/", "-")


def fetch_one(ticker: str, now: dt.datetime, tries: int = 3) -> pd.DataFrame:
    import yfinance as yf
    last_err = None
    for k in range(tries):
        try:
            tk = yf.Ticker(yahoo_symbol(ticker))
            h = tk.history(period="5d", auto_adjust=False)
            if h.empty:
                raise ValueError("no price data")
            bar_day = pd.Timestamp(h.index[-1]).date()
            spot = float(h["Close"].iloc[-1])
            exps = list(tk.options or [])
            if not exps:
                raise ValueError("no listed options")
            frames = []
            for e in exps:
                ch = tk.option_chain(e)
                exp_close = dt.datetime.combine(dt.date.fromisoformat(e), dt.time(16, 0), NY)
                T = (exp_close - now).total_seconds() / (365 * 86400)
                for is_call, fr in ((True, ch.calls), (False, ch.puts)):
                    if fr is None or fr.empty:
                        continue
                    frames.append(pd.DataFrame({
                        "strike": fr["strike"].astype(float), "is_call": is_call,
                        "iv": fr["impliedVolatility"].astype(float),
                        "oi": pd.to_numeric(fr["openInterest"], errors="coerce").fillna(0).astype(float),
                        "volume": pd.to_numeric(fr["volume"], errors="coerce").fillna(0).astype(float),
                        "expiry_str": e, "T": T}))
            df = pd.concat(frames, ignore_index=True)
            df["date"] = now.date(); df["ticker"] = ticker; df["spot"] = spot; df["source"] = "yahoo"
            df.attrs["bar_day"] = bar_day
            return df
        except Exception as ex:          # network hiccups, rate limits
            last_err = ex
            time.sleep(3 * (k + 1))
    raise RuntimeError(str(last_err))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    now = dt.datetime.now(NY)
    today = now.date()
    if not a.force:
        if not is_trading_day(today):
            print(f"{today} is not a trading day - nothing to do."); return 0
        if now.time() < EARLIEST:
            print(f"It is {now:%H:%M} in New York; the update runs after {EARLIEST:%H:%M}."); return 0
        s = OS.read_summary()
        if len(s) and (s["date"].dt.date == today).any() and s.loc[s["date"].dt.date == today, "pc_volume"].notna().sum() > 0:
            print(f"{today} is already collected - nothing to do."); return 0

    if not a.tickers and not OS.universe():
        print("options_data/universe.txt not found - run the one-time import first (see README). Nothing to do.")
        return 0
    tickers = [t.upper() for t in (a.tickers or OS.universe())]
    print(f"Collecting {len(tickers)} tickers for {today} ...")
    t0 = time.time(); ok, failed, stale = [], [], []
    frames = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(fetch_one, t, now): t for t in tickers}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                df = f.result()
                if df.attrs.get("bar_day") != today:
                    stale.append(t)
                frames.append(df); ok.append(t)
            except Exception as e:
                failed.append((t, str(e)[:120]))
            if i % 50 == 0:
                print(f"  {i}/{len(tickers)} done ({len(failed)} failed, {time.time() - t0:.0f}s)")
    if not frames:
        print("No data collected."); return 1
    day = pd.concat(frames, ignore_index=True)
    res = OS.add_chain_day(day)
    m = OS.load_manifest()
    summ = OS.summarize(OS.normalize_chain(day, "yahoo"), gex_scale=m.get("gex_scale", 1.0))
    OS.upsert_summary(summ)
    m["last_update"] = now.isoformat(timespec="minutes")
    OS.save_manifest(m)
    log = OS.paths()["root"] / "update_log.csv"
    row = pd.DataFrame([{"run_at": now.isoformat(timespec="minutes"), "date": today, "tickers_ok": len(ok),
                         "tickers_failed": len(failed), "stale_prices": len(stale), "contracts": len(day),
                         "seconds": round(time.time() - t0), "compacted": res.get("compacted", False)}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"Stored {len(day):,} contracts for {len(ok)} tickers in {time.time() - t0:.0f}s; "
          f"{len(failed)} failed; {len(stale)} with an old price bar.")
    for t, e in failed[:30]:
        print(f"  FAILED {t}: {e}")
    if res.get("compacted"):
        print(f"Merged pending days into {[p['file'] for p in res['new_parts']]}")
    return 0 if len(ok) >= 0.5 * len(tickers) else 1


if __name__ == "__main__":
    sys.exit(main())
