"""Framework 1 - data cleaning (book ch. 2: logical and statistical editing)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE trading days if pandas_market_calendars is installed, else weekdays."""
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("NYSE")
        sched = cal.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index).normalize().tz_localize(None)
    except Exception:
        return pd.bdate_range(start, end)


def clean(raw: pd.DataFrame, s: dict, calendar: pd.DatetimeIndex | None = None):
    """Return (clean_df, log, quality).

    clean_df has Open, High, Low, Close, Volume and two extra columns:
      ret      - daily simple return of the cleaned close
      ret_win  - the same return, winsorized at the rolling 3-sigma limits
                 (used by the statistical tests; prices are NOT changed)
    """
    log: list[dict] = []
    df = raw[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    def note(date, check, action, detail=""):
        log.append({"date": pd.Timestamp(date).date(), "check": check, "action": action, "detail": detail})

    # --- 1. Logical checks -------------------------------------------------
    cal = calendar if calendar is not None else trading_days(df.index[0], df.index[-1])
    closed = df.index.difference(cal)
    for d in closed:
        note(d, "Market closed that day", "Row removed")
    df = df.drop(closed)

    neg_v = df["Volume"] < 0
    for d in df.index[neg_v]:
        note(d, "Negative volume", "Volume set to missing")
    df.loc[neg_v, "Volume"] = np.nan

    o, h, l, c = (df[k] for k in ("Open", "High", "Low", "Close"))
    bad = (
        (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
        | (h < l)
        | (c > h) | (c < l)
        | (o > h) | (o < l)
    )
    for d in df.index[bad.fillna(False)]:
        r = df.loc[d]
        note(d, "High/low/open/close inconsistent", "Prices set to missing",
             f"O={r.Open:.2f} H={r.High:.2f} L={r.Low:.2f} C={r.Close:.2f}")
    df.loc[bad.fillna(False), ["Open", "High", "Low", "Close"]] = np.nan

    # bad quote: a big one-day move that is fully reversed the next day
    mv = float(s["bad_quote_move"])
    cc = df["Close"]
    prev, nxt = cc.shift(1), cc.shift(-1)
    jump = (cc / prev - 1).abs() > mv
    back = (nxt / prev - 1).abs() < mv / 5
    bq = (jump & back).fillna(False)
    for d in df.index[bq]:
        note(d, "Bad quote (big move reversed next day)", "Prices set to missing",
             f"close {cc[d]:.2f} between {prev[d]:.2f} and {nxt[d]:.2f}")
    df.loc[bq, ["Open", "High", "Low", "Close"]] = np.nan

    big = ((cc / prev - 1).abs() > 0.40) & ~bq
    for d in df.index[big.fillna(False)]:
        note(d, "One-day move above 40%", "Kept - please check (possible missing split adjustment)",
             f"{(cc[d] / prev[d] - 1):+.1%}")

    # --- 3. Missing values -------------------------------------------------
    full = cal[(cal >= df.index[0]) & (cal <= df.index[-1])]
    missing_days = full.difference(df.index)
    df = df.reindex(full)
    miss = df["Close"].isna()
    yearly = miss.groupby(df.index.year).mean()
    worst_year = int(yearly.idxmax()) if len(yearly) else None
    worst_share = float(yearly.max()) if len(yearly) else 0.0
    for d in missing_days:
        note(d, "Missing trading day", "Filled by straight-line interpolation")
    n_interp = int(miss.sum())
    df[["Open", "High", "Low", "Close", "Volume"]] = (
        df[["Open", "High", "Low", "Close", "Volume"]].interpolate(method="linear", limit_area="inside"))
    df = df.dropna(subset=["Close"])
    for k in ("Open", "High", "Low"):
        df[k] = df[k].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    # interpolation can break H >= max(O, C) slightly; repair
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    # --- 2. Statistical check on returns (rolling, trailing window) ---------
    N = int(s["clean_window"]); k = float(s["clean_sigma"])
    r = df["Close"].pct_change()
    mu = r.rolling(N, min_periods=N).mean()
    sd = r.rolling(N, min_periods=N).std(ddof=1)
    lo, hi = mu - k * sd, mu + k * sd
    flag = ((r < lo) | (r > hi)).fillna(False)
    df["ret"] = r
    df["ret_win"] = r.clip(lower=lo, upper=hi).where(flag, r)
    n_flag = int(flag.sum())
    for d in df.index[flag]:
        note(d, f"Return beyond {k:.0f} standard deviations", "Kept in prices; winsorized copy used in tests",
             f"{r[d]:+.2%}")

    quality = {
        "passed": worst_share <= float(s["max_missing_share"]),
        "worst_year": worst_year, "worst_year_missing": worst_share,
        "n_days": len(df), "n_interpolated": n_interp, "n_extreme": n_flag,
        "n_logical": int(bad.sum() + bq.sum() + neg_v.sum() + len(closed)),
        "first": df.index[0].date() if len(df) else None, "last": df.index[-1].date() if len(df) else None,
    }
    return df, log, quality
