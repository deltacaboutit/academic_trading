"""Daily option-chain collector: downloads every stock's chain from Yahoo Finance
after the close and adds it to this repo's options store. Replaces the copy from
Watchtower; it collects the same contracts, the same way.

    python -m atrade.collect_chains                 # today's session (or the last one, before 4:15 pm)
    python -m atrade.collect_chains --check         # only print how many stocks still need the session
    python -m atrade.collect_chains --tickers AAPL SPY --force

What is collected for each stock (as Watchtower did):
  - every expiry from the session day up to 90 calendar days out,
  - every call and put with open interest >= 5 and a reported implied volatility,
  - strike, IV, open interest, volume, bid, ask, last price, and the session's closing price (spot).
T (years to expiry) is measured from the session's 4:00 pm close to the expiry's 4:00 pm close,
so it does not depend on when the collector happens to run (tool choice).

Session day: the NYSE trading day whose close has passed (4:15 pm New York time or later),
otherwise the previous trading day. A stock that already has the session is skipped, so the
evening run and the catch-up runs only fetch what is still missing.

Then, as for every other source: junk IVs are filled (options_store.clean_iv), the day is appended
to the stock's file, and the summary gets put/call ratios, average IV and the GEX levels
(gex.daily_levels, Watchtower's method, using FRED's 3-month T-bill rate).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import gex as G
from . import options_store as OS

NY = ZoneInfo("America/New_York")
MAX_DAYS = 90            # expiries up to 90 days out (Watchtower's setting)
MIN_OI = 5               # contracts with open interest below 5 are dropped (Watchtower's setting)
CLOSE_DONE = dt.time(16, 15)
PAUSE = 1.0              # seconds between stocks, to stay under Yahoo's rate limit (Watchtower used 2)
TRIES = 3
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
DEFAULT_RATE = 0.045
YEAR_SECONDS = 365 * 24 * 3600


# ---------------------------------------------------------------- dates ---

def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    try:
        import pandas_market_calendars as mcal
        return [d.date() for d in mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end).index]
    except Exception:
        return [d.date() for d in pd.bdate_range(start, end)]


def session_day(now: dt.datetime | None = None) -> dt.date:
    """The latest NYSE session whose close (plus 15 minutes) has passed."""
    now = (now or dt.datetime.now(NY)).astimezone(NY)
    days = trading_days(now.date() - dt.timedelta(days=12), now.date())
    if days and days[-1] == now.date() and now.time() < CLOSE_DONE:
        days = days[:-1]
    return days[-1]


def market_open_now(now: dt.datetime | None = None) -> bool:
    """True from 9:25 am to 4:15 pm New York time on a trading day. Yahoo's chain is then
    the live, half-finished day, which must not be saved as the previous session."""
    now = (now or dt.datetime.now(NY)).astimezone(NY)
    if not (dt.time(9, 25) <= now.time() < CLOSE_DONE):
        return False
    return now.date() in trading_days(now.date(), now.date())


def years_to_expiry(expiry: str, session: dt.date) -> float:
    close = dt.datetime.combine(session, dt.time(16), NY)
    exp = dt.datetime.combine(dt.date.fromisoformat(expiry), dt.time(16), NY)
    return max((exp - close).total_seconds(), 60) / YEAR_SECONDS


# ----------------------------------------------------------------- rate ---

def risk_free_rate(session: dt.date, store=None) -> float:
    """FRED 3-month T-bill rate on or before the session (cached in options_data/rate.json)."""
    p = OS.paths(store)["root"] / "rate.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    try:
        import io

        import requests
        raw = pd.read_csv(io.StringIO(requests.get(FRED_URL, timeout=20).text))
        raw.columns = ["date", "rate"]
        raw["rate"] = pd.to_numeric(raw["rate"], errors="coerce") / 100
        raw = raw.dropna()
        raw = raw[pd.to_datetime(raw["date"]).dt.date <= session]
        if len(raw):
            cache = {"date": str(raw["date"].iloc[-1]), "rate": float(raw["rate"].iloc[-1])}
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cache))
    except Exception as e:
        print(f"FRED rate not available ({e}); using {'the saved rate' if cache else 'the default 4.5%'}.")
    return float(cache.get("rate", DEFAULT_RATE))


# --------------------------------------------------------------- yahoo ---

def fetch_stock(ticker: str, session: dt.date, yf=None) -> pd.DataFrame:
    """One stock's chain for the session, in Yahoo's column names plus is_call/expiry_str/spot/T.
    Raises if Yahoo returns nothing usable."""
    if yf is None:
        import yfinance as yf
    t = yf.Ticker(ticker)
    h = t.history(period="10d", auto_adjust=False)
    if h is None or h.empty:
        raise ValueError("no price history")
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    if h.index[-1].date() < session:
        raise ValueError(f"Yahoo's latest close is {h.index[-1].date()}, not {session} yet")
    spot = float(h.loc[h.index.date == session, "Close"].iloc[-1]) if (h.index.date == session).any() \
        else float(h["Close"].iloc[-1])
    exps = [e for e in (t.options or []) if 0 <= (dt.date.fromisoformat(e) - session).days <= MAX_DAYS]
    if not exps:
        raise ValueError("no option expiries within 90 days")
    parts = []
    for e in exps:
        ch = t.option_chain(e)
        for df, is_call in ((ch.calls, True), (ch.puts, False)):
            if df is not None and len(df):
                parts.append(df.assign(is_call=is_call, expiry_str=e))
    if not parts:
        raise ValueError("empty chain")
    df = pd.concat(parts, ignore_index=True).rename(columns={"impliedVolatility": "iv", "openInterest": "oi"})
    df = df[df["oi"] >= MIN_OI].dropna(subset=["iv", "oi"])
    if df.empty:
        raise ValueError("no contracts with open interest >= 5")
    df["T"] = df["expiry_str"].map(lambda e: years_to_expiry(e, session))
    df["spot"] = spot
    return df


def fetch_with_retries(ticker, session, yf=None, tries=TRIES, sleep=time.sleep):
    err = None
    for k in range(tries):
        try:
            return fetch_stock(ticker, session, yf), None
        except Exception as e:                              # Yahoo throttling, timeouts, missing data
            err = f"{type(e).__name__}: {e}"
            if "not " in str(e) and "yet" in str(e):        # session not published yet: no point retrying now
                break
            sleep(5 * (k + 1) ** 2)
    return None, err


# ------------------------------------------------------------ collector ---

def already_have(session: dt.date, store=None) -> set[str]:
    s = OS.read_summary(store=store)
    if s.empty:
        return set()
    return set(s.loc[s["date"].dt.date == session, "ticker"])


def collect(tickers=None, session=None, force=False, store=None, chain_dir=None, yf=None,
            pause=PAUSE, sleep=time.sleep, log=print) -> dict:
    session = session or session_day()
    uni = [t.upper() for t in (tickers or OS.universe(store))]
    todo = uni if force else [t for t in uni if t not in already_have(session, store)]
    log(f"Session {session}: {len(uni)} stocks in the list, {len(todo)} still to collect.")
    if todo and yf is None and market_open_now():
        log("The market is open: Yahoo's chains are the live day, not a finished session. "
            "Nothing collected; the evening run will do it.")
        return {"session": str(session), "todo": len(todo), "ok": 0, "failed": {}, "market_open": True}
    st_path = OS.paths(store)["root"] / "collect_state.json"
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if not todo:
        return {"session": str(session), "todo": 0, "ok": 0, "failed": {}}
    rate = risk_free_rate(session, store)
    log(f"Risk-free rate (3-month T-bill): {rate:.2%}")
    ok, failed, summ = [], {}, []
    started = time.time()

    def save(ticker, raw):
        raw = raw.assign(ticker=ticker, date=session, source="yahoo")
        ch = OS.clean_iv(OS.normalize_chain(raw, "yahoo"))
        OS.append_stock(ticker, ch, chain_dir=chain_dir, keep="new")
        summ.append(OS.summarize(ch, compute_gex=True, rate=rate))
        ok.append(ticker)
        return len(ch)

    for i, t in enumerate(todo):
        if yf is None and market_open_now():
            log(f"The market opened; stopping. {len(todo) - i} stocks are left for the evening run.")
            break
        raw, err = fetch_with_retries(t, session, yf, sleep=sleep)
        if raw is None:
            failed[t] = err
            log(f"  {t}: FAILED - {err}")
        else:
            log(f"  {t}: {save(t, raw):,} contracts")
        if i + 1 < len(todo) and pause:
            sleep(pause)
        if summ and len(summ) % 50 == 0:                    # save progress in case the job is stopped
            OS.upsert_summary(pd.concat(summ, ignore_index=True), store); summ = []
    if failed and not (yf is None and market_open_now()):    # one more pass for stocks that failed
        log(f"Second try for {len(failed)} stocks after a 60-second pause ...")
        sleep(60)
        for t in list(failed):
            raw, err = fetch_with_retries(t, session, yf, sleep=sleep)
            if raw is not None:
                del failed[t]
                log(f"  {t}: {save(t, raw):,} contracts (second try)")
            else:
                failed[t] = err
    if summ:
        OS.upsert_summary(pd.concat(summ, ignore_index=True), store)
    P = OS.paths(store)
    if ok:
        P["universe"].parent.mkdir(parents=True, exist_ok=True)
        P["universe"].write_text("\n".join(sorted(set(uni) | set(ok))) + "\n")
    have = already_have(session, store) & set(uni)
    state[str(session)] = {"stocks": len(uni), "collected": len(have), "failed": failed,
                           "last_run": dt.datetime.now(NY).isoformat(timespec="minutes"),
                           "minutes": round((time.time() - started) / 60, 1)}
    state = dict(sorted(state.items())[-30:])                # keep the last 30 sessions
    st_path.parent.mkdir(parents=True, exist_ok=True)
    st_path.write_text(json.dumps(state, indent=1))
    log(f"Session {session}: {len(have)} of {len(uni)} stocks collected "
        f"({len(ok)} this run, {len(failed)} failed) in {state[str(session)]['minutes']} minutes.")
    return {"session": str(session), "todo": len(todo), "ok": len(ok), "failed": failed,
            "collected": len(have), "stocks": len(uni)}


def fred_rates() -> pd.Series:
    """The whole FRED 3-month T-bill series (decimal), indexed by date."""
    import io

    import requests
    raw = pd.read_csv(io.StringIO(requests.get(FRED_URL, timeout=30).text))
    raw.columns = ["date", "rate"]
    raw["rate"] = pd.to_numeric(raw["rate"], errors="coerce") / 100
    return raw.dropna().set_index(pd.to_datetime(raw.dropna()["date"]))["rate"]


def recompute_walls(store=None, chain_dir=None, rates_file=None, log=print) -> int:
    """Put the walls of every past summary day at strikes (README "GEX levels"), using the
    stored chain of that stock and day. Flip and total GEX are left as they are."""
    P = OS.paths(store)
    try:
        if rates_file:                                   # a saved DGS3MO series (columns date, rate)
            rf = pd.read_parquet(rates_file) if str(rates_file).endswith(".parquet") else pd.read_csv(rates_file)
            rates = rf.set_index(pd.to_datetime(rf["date"]))["rate"].astype(float).sort_index()
        else:
            rates = fred_rates()
    except Exception as e:
        log(f"FRED not reachable ({e}); using 4.5% for every day.")
        rates = pd.Series(dtype=float)

    def rate_on(d):
        r = rates[rates.index <= pd.Timestamp(d)]
        return float(r.iloc[-1]) if len(r) else DEFAULT_RATE

    files = sorted(P["summary"].glob("*.parquet"))
    summ = pd.concat([pd.read_parquet(f).assign(_file=f.name) for f in files], ignore_index=True)
    for c in OS.SUMMARY_COLS:
        if c not in summ:
            summ[c] = np.nan
    summ["date"] = pd.to_datetime(summ["date"]).dt.date
    summ["wall_method"] = summ["wall_method"].astype(object)
    todo = summ[summ["wall_method"].astype(str) != "strike"]
    log(f"{len(todo):,} stock-days to recompute ({todo['ticker'].nunique()} stocks).")
    cols = ["call_wall_price", "call_wall_gex", "put_wall_price", "put_wall_gex"]
    done = 0
    for i, (t, rows) in enumerate(todo.groupby("ticker")):
        try:
            ch = OS.read_chain(t, chain_dir=chain_dir)
        except Exception as e:
            log(f"  {t}: chain not available ({e})"); continue
        ch["date"] = pd.to_datetime(ch["date"]).dt.date
        by_day = dict(tuple(ch.groupby("date")))
        for ix, row in rows.iterrows():
            g = by_day.get(row["date"])
            if g is None or not np.isfinite(row["spot"]):
                continue
            w = G.strike_walls(G._filtered(g), float(row["spot"]), rate_on(row["date"]))
            for c in cols:
                summ.at[ix, c] = w[c]
            summ.at[ix, "wall_method"] = "strike"
            done += 1
        if (i + 1) % 50 == 0:
            log(f"  {i + 1} stocks done")
    for name, part in summ.groupby("_file"):
        part = part.drop(columns="_file")[OS.SUMMARY_COLS].sort_values(["date", "ticker"]).reset_index(drop=True)
        part.to_parquet(P["summary"] / name, index=False, compression="zstd")
    log(f"Walls recomputed at strikes for {done:,} stock-days.")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--session", help="YYYY-MM-DD (default: the latest finished session)")
    ap.add_argument("--force", action="store_true", help="collect even stocks that already have the session")
    ap.add_argument("--check", action="store_true", help="only print how many stocks still need the session")
    ap.add_argument("--min-share", type=float, default=0.95,
                    help="with --check-complete: fail unless this share of stocks has the session")
    ap.add_argument("--check-complete", action="store_true")
    ap.add_argument("--recompute-walls", action="store_true",
                    help="one-off: put the walls of all past summary days at strikes")
    ap.add_argument("--rates-file", help="with --recompute-walls: saved FRED DGS3MO series instead of FRED")
    a = ap.parse_args(argv)
    if a.recompute_walls:
        recompute_walls(rates_file=a.rates_file)
        return 0
    session = dt.date.fromisoformat(a.session) if a.session else session_day()
    uni = [t.upper() for t in (a.tickers or OS.universe())]
    if a.check and market_open_now():
        print("todo=0")
        print("The market is open; the chains will be collected after the close.")
        return 0
    if a.check or a.check_complete:
        have = already_have(session) & set(uni)
        print(f"todo={len(uni) - len(have)}")
        print(f"Session {session}: {len(have)} of {len(uni)} stocks collected.")
        if a.check_complete and len(have) < a.min_share * len(uni):
            missing = sorted(set(uni) - have)
            print(f"Missing: {' '.join(missing[:60])}{' ...' if len(missing) > 60 else ''}")
            return 1
        return 0
    collect(uni, session, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
