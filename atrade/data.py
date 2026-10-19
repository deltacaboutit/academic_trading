"""Data loading: Yahoo Finance prices and VIX, daily options snapshots, CSV imports."""
from __future__ import annotations

import datetime as dt
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
        raise


def load_vix(years: int) -> pd.Series | None:
    try:
        df, _ = load_prices("^VIX", years)
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


def load_sentiment(ticker: str, dates: pd.DatetimeIndex, years: int) -> dict:
    """All sentiment series aligned to the price dates (NaN where missing)."""
    ticker = ticker.upper()
    out: dict[str, np.ndarray] = {}
    if ticker == "DEMO":
        from .demo import demo_sentiment
        return demo_sentiment(dates)

    def align(sr):
        return sr.reindex(dates).to_numpy(float) if sr is not None and len(sr) else None

    vix = load_vix(years)
    out["vix"] = align(vix)
    out["equity_pc"] = align(_read_import("equity_pc"))
    snap_path = SENTIMENT_DIR / f"{ticker}_options.csv"
    snap = pd.read_csv(snap_path, parse_dates=["date"]).set_index("date") if snap_path.exists() else None
    for key, col in (("stock_pc", "put_call_ratio"), ("stock_iv", "avg_iv")):
        imp = _read_import(f"{ticker}_{key.split('_')[1]}")
        sr = imp
        if snap is not None and col in snap:
            s2 = snap[col].dropna()
            sr = s2 if sr is None else pd.concat([sr[~sr.index.isin(s2.index)], s2]).sort_index()
        out[key] = align(sr)
    return {k: v for k, v in out.items() if v is not None}
