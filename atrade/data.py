"""Data loading: Yahoo Finance prices and VIX, daily options snapshots, CSV imports."""
from __future__ import annotations

import datetime as dt
import os
import io
import time

import numpy as np
import pandas as pd

from .config import CACHE_DIR, SENTIMENT_DIR

CBOE_EQUITY_PC_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"


def _ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


def _yahoo_history(symbol: str, years: int) -> pd.DataFrame:
    import yfinance as yf
    start = (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 5)).isoformat()
    h = yf.Ticker(symbol).history(start=start, auto_adjust=True, actions=False)
    if h is None or h.empty:
        raise ValueError(f"Yahoo Finance returned no data for {symbol}.")
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    return h[["Open", "High", "Low", "Close", "Volume"]]


def load_prices(ticker: str, years: int, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Daily prices adjusted for splits and dividends. Returns (df, source note).

    Uses a local copy if it was saved today, so repeated analyses are fast and
    work offline.
    """
    _ensure_dirs()
    ticker = ticker.upper().strip()
    if ticker == "DEMO":
        from .demo import demo_prices
        return demo_prices(), "Built-in demo data (simulated prices, not a real stock)"
    if os.environ.get("ATRADE_OFFLINE"):          # research run: use price_data/ only, no downloads
        repo = repo_prices(ticker)
        if repo is not None:
            return repo, f"price_data/ in the repo (to {repo.index[-1].date()})"
        raise ValueError(f"{ticker} is not in price_data/.")
    path = CACHE_DIR / f"{ticker.replace('^', '_')}.csv"
    fresh = path.exists() and dt.date.fromtimestamp(path.stat().st_mtime) == dt.date.today()
    if fresh and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df, f"Yahoo Finance (saved copy from today, {len(df)} days)"
    try:
        df = _yahoo_history(ticker, years)
        df.to_csv(path)
        return df, f"Yahoo Finance (downloaded {dt.datetime.now():%Y-%m-%d %H:%M}, {len(df)} days)"
    except Exception as e:
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df, f"Saved copy from {dt.date.fromtimestamp(path.stat().st_mtime)} (download failed: {e})"
        repo = repo_prices(ticker)
        if repo is not None:
            return repo, f"price_data/ in the repo (to {repo.index[-1].date()}; Yahoo Finance could not be reached)"
        raise


def repo_prices(ticker: str) -> pd.DataFrame | None:
    """Price history saved by 'python -m atrade.download_prices' (price_data/ in the repo)."""
    from .config import ROOT
    p = ROOT / "price_data" / (ticker.upper().replace("^", "_").replace("/", "-") + ".parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_vix(years: int) -> pd.Series | None:
    try:
        df, _ = load_prices("^VIX", years)
        return df["Close"]
    except Exception:
        return None


# ------------------------------------------- other bar sizes and CSV files ---
# Added from PatternFinder 2.5 (pattern_core/data.py and streamlit_app.py).

BAR_UNITS = {"day": "trading day", "week": "week", "hour": "hour"}
HOURLY_DAYS = 729       # Yahoo Finance keeps hourly prices for about the last two years only


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars combined into weekly bars ending on Friday: the week's first open, highest
    high, lowest low, last close and total volume. A week is dated by its last trading day."""
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    wk = d.index.to_period("W-FRI")
    g = d.groupby(wk)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    out.index = pd.DatetimeIndex(d.index.to_series().groupby(wk).max().to_numpy())
    return out.dropna(subset=["Close"])


def load_hourly(ticker: str, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Hourly bars for about the last two years from Yahoo Finance (a saved copy is reused on the same day)."""
    _ensure_dirs()
    ticker = ticker.upper().strip()
    if ticker == "DEMO":
        raise ValueError("Hourly bars are not available for the demo data. Pick Daily or Weekly.")
    path = CACHE_DIR / f"{ticker.replace('^', '_')}_1h.csv"
    if path.exists() and not refresh and dt.date.fromtimestamp(path.stat().st_mtime) == dt.date.today():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df, f"Yahoo Finance hourly bars (saved copy from today, {len(df)} hours)"
    import yfinance as yf
    raw = yf.Ticker(ticker).history(period=f"{HOURLY_DAYS}d", interval="1h", auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no hourly prices for {ticker}.")
    raw.index = pd.DatetimeIndex(raw.index).tz_convert("America/New_York").tz_localize(None)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.to_csv(path)
    return df, f"Yahoo Finance hourly bars (downloaded {dt.datetime.now():%Y-%m-%d %H:%M}, {len(df)} hours)"


def parse_price_csv(content: bytes) -> pd.DataFrame:
    """Read a price file you saved yourself (PatternFinder 2.5's load_csv).

    It needs a date column (named Date, Datetime, Time or Timestamp) and the columns
    Open, High, Low, Close and, if available, Volume. Upper or lower case does not matter.
    """
    import io
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Could not read the file: {e}") from e
    cols = ["Open", "High", "Low", "Close", "Volume"]
    rename = {}
    for col in raw.columns:
        key = str(col).strip().lower().replace(" ", "").replace("_", "")
        if "." in key:                                  # e.g. "AAPL.Close" -> "close"
            key = key.split(".")[-1]
        if key in ("date", "datetime", "time", "timestamp"):
            rename[col] = "Date"
        else:
            for want in cols:
                if key == want.lower():
                    rename[col] = want
    raw = raw.rename(columns=rename)
    if "Date" not in raw.columns:
        raise ValueError("The CSV file needs a column called 'Date'.")
    missing = [c for c in cols[:4] if c not in raw.columns]
    if missing:
        raise ValueError("The CSV file is missing these columns: " + ", ".join(missing))
    if "Volume" not in raw.columns:
        raw["Volume"] = 0.0
    dates = raw["Date"].astype(str)
    if dates.str.contains(r"[+-]\d\d:\d\d$").any():   # dates with a time-zone offset: keep the local time
        raw["Date"] = pd.to_datetime(dates.str.slice(0, 19), errors="coerce")
    else:
        raw["Date"] = pd.to_datetime(dates, errors="coerce")
    df = raw.dropna(subset=["Date"]).set_index("Date")[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise ValueError("The CSV file has no usable rows.")
    return df


def bar_date(index, i: int, bar: str = "day"):
    """The date of bar number i. Bars after the last one are counted forward in trading days,
    weeks (Fridays) or trading hours (about 7 a day), to match the bar size."""
    idx = pd.DatetimeIndex(index)
    if 0 <= i < len(idx):
        return idx[i]
    k = i - (len(idx) - 1)
    if bar == "week":                                  # the Friday of the last bar's week, then k weeks on
        friday = idx[-1].normalize() + pd.Timedelta(days=(4 - idx[-1].weekday()) % 7)
        return friday + pd.Timedelta(weeks=k)
    if bar == "hour":
        return (idx[-1] + pd.offsets.BDay(-(-k // 7))).normalize()
    return idx[-1] + pd.offsets.BDay(k)


def is_intraday(index) -> bool:
    """True if the dates include times of day (for example hourly bars)."""
    idx = pd.DatetimeIndex(index)
    return bool(len(idx) and (idx != idx.normalize()).any())


def load_market(years: int) -> pd.Series | None:
    """SPY's closing prices, for the market and relative-strength checks (confirm.py)."""
    try:
        df, _ = load_prices("SPY", years)
        return df["Close"]
    except Exception:
        return None


# --------------------------------------------------- options snapshots ------

def snapshot_options(ticker: str) -> dict:
    """Save today's total put and call volume and average implied volatility for
    one stock. Run once a day after the market closes (4:30 pm New York time or
    later) so the day's volume is complete.
    """
    import yfinance as yf
    _ensure_dirs()
    ticker = ticker.upper().strip()
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError(f"No recent prices for {ticker}.")
    day = pd.Timestamp(hist.index[-1]).tz_localize(None).normalize()
    put_v = call_v = 0.0
    ivs = []
    exps = list(tk.options or [])
    if not exps:
        raise ValueError(f"{ticker} has no listed options on Yahoo Finance.")
    for e in exps:
        try:
            ch = tk.option_chain(e)
        except Exception:
            time.sleep(0.5)
            continue
        for side, frame in (("call", ch.calls), ("put", ch.puts)):
            v = frame["volume"].fillna(0).astype(float)
            if side == "put":
                put_v += v.sum()
            else:
                call_v += v.sum()
            iv = frame["impliedVolatility"].astype(float)
            m = (v > 0) & iv.notna() & (iv > 0.01) & (iv < 5)
            ivs.extend(iv[m].tolist())
    row = {
        "date": day.date().isoformat(),
        "put_volume": put_v, "call_volume": call_v,
        "put_call_ratio": put_v / call_v if call_v > 0 else np.nan,
        "avg_iv": float(np.mean(ivs)) if ivs else np.nan,
        "expirations": len(exps),
        "captured_at": dt.datetime.now().isoformat(timespec="minutes"),
    }
    path = SENTIMENT_DIR / f"{ticker}_options.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    old = old[old["date"] != row["date"]]
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    new.to_csv(path, index=False)
    return row


def snapshot_count(ticker: str) -> int:
    path = SENTIMENT_DIR / f"{ticker.upper()}_options.csv"
    return len(pd.read_csv(path)) if path.exists() else 0


# ------------------------------------------------------------ imports -------

def parse_series_csv(content: bytes) -> pd.Series:
    """Read a CSV with a date column and a value column (first date-like column,
    last numeric column). Lines before the header row are skipped."""
    text = content.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines[:50]):
        low = ln.lower()
        if "date" in low and "," in ln:
            start = i
            break
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    date_col = None
    for c in df.columns:
        try:
            pd.to_datetime(df[c].head(20), errors="raise")
            date_col = c
            break
        except Exception:
            continue
    if date_col is None:
        raise ValueError("Could not find a date column.")
    num = [c for c in df.columns if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    if not num:
        raise ValueError("Could not find a numeric value column.")
    val = num[-1]
    for c in num:
        if "p/c" in str(c).lower() or "ratio" in str(c).lower() or "iv" == str(c).lower().strip():
            val = c
    sr = pd.Series(pd.to_numeric(df[val], errors="coerce").to_numpy(),
                   index=pd.to_datetime(df[date_col], errors="coerce")).dropna()
    sr = sr[~sr.index.isna()].sort_index()
    sr.index = sr.index.normalize()
    return sr[~sr.index.duplicated(keep="last")]


def save_import(key: str, sr: pd.Series):
    """key: 'equity_pc', or '<TICKER>_pc' / '<TICKER>_iv'."""
    _ensure_dirs()
    path = SENTIMENT_DIR / f"import_{key}.csv"
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        sr = pd.concat([old[~old.index.isin(sr.index)], sr]).sort_index()
    sr.rename("value").to_csv(path)
    return len(sr)


def try_download_cboe_equity_pc() -> str:
    """Cboe's free archive of the equity-only put/call ratio (ends in 2019)."""
    import requests
    r = requests.get(CBOE_EQUITY_PC_URL, timeout=20)
    r.raise_for_status()
    sr = parse_series_csv(r.content)
    n = save_import("equity_pc", sr)
    return f"Saved {n} days of the equity-only put/call ratio ({sr.index[0].date()} to {sr.index[-1].date()})."


def _read_import(key: str) -> pd.Series | None:
    path = SENTIMENT_DIR / f"import_{key}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def _store_summary(ticker: str):
    try:
        from . import options_store as OS
        s = OS.read_summary([ticker])
        return s.set_index("date") if len(s) else None
    except Exception:
        return None


def _market_pc():
    try:
        from . import options_store as OS
        m = OS.market_put_call()
        return m if len(m) else None
    except Exception:
        return None


def load_sentiment(ticker: str, dates: pd.DatetimeIndex, years: int) -> dict:
    """All sentiment series aligned to the price dates (NaN where missing).

    stock_pc      the stock's put/call ratio from VOLUME (daily store, snapshots, imports)
    stock_pc_oi   the stock's put/call ratio from OPEN INTEREST (labelled substitute)
    stock_iv      the stock's average implied volatility
    equity_pc     equity-only put/call (Cboe import, else market-wide volume ratio)
    equity_pc_oi  market-wide put/call from open interest (labelled substitute)
    vix           CBOE Volatility Index
    """
    ticker = ticker.upper()
    out: dict[str, np.ndarray] = {}
    if ticker == "DEMO":
        from .demo import demo_sentiment
        return demo_sentiment(dates)

    def align(sr):
        if sr is None:
            return None
        sr = sr.dropna()
        if not len(sr):
            return None
        sr.index = pd.DatetimeIndex(sr.index).normalize()
        sr = sr[~sr.index.duplicated(keep="last")]
        return sr.reindex(dates).to_numpy(float)

    def combine(*parts):
        sr = None
        for p in parts:
            if p is None or not len(p.dropna()):
                continue
            p = p.dropna(); p.index = pd.DatetimeIndex(p.index).normalize()
            sr = p if sr is None else sr.combine_first(p)
        return sr

    out["vix"] = align(load_vix(years))
    store = _store_summary(ticker)
    mkt = _market_pc()

    # market-wide / equity-only put/call
    out["equity_pc"] = align(combine(mkt["pc_volume"] if mkt is not None else None, _read_import("equity_pc")))
    out["equity_pc_oi"] = align(mkt["pc_oi"] if mkt is not None else None)

    # the stock's own ratios: store first, then local snapshots and imports
    snap_path = SENTIMENT_DIR / f"{ticker}_options.csv"
    snap = pd.read_csv(snap_path, parse_dates=["date"]).set_index("date") if snap_path.exists() else None
    out["stock_pc"] = align(combine(store["pc_volume"] if store is not None else None,
                                    snap["put_call_ratio"] if snap is not None else None,
                                    _read_import(f"{ticker}_pc")))
    out["stock_pc_oi"] = align(store["pc_oi"] if store is not None else None)
    out["stock_iv"] = align(combine(store["avg_iv"] if store is not None else None,
                                    snap["avg_iv"] if snap is not None else None,
                                    _read_import(f"{ticker}_iv")))
    return {k: v for k, v in out.items() if v is not None}


def load_gex(ticker: str, dates: pd.DatetimeIndex) -> dict | None:
    """[GEX] flip and walls aligned to the price dates, or None if not available."""
    s = _store_summary(ticker.upper())
    if s is None or s["gex_flip"].notna().sum() + s["call_wall_price"].notna().sum() == 0:
        return None
    s = s[~s.index.duplicated(keep="last")]
    cols = ["gex_flip", "call_wall_price", "call_wall_gex", "put_wall_price", "put_wall_gex"]
    return {c: s[c].astype(float).reindex(dates).to_numpy(float) for c in cols}
