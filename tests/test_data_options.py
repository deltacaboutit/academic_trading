"""Checks of weekly bars, hourly bars and loading a CSV price file (data options from PatternFinder 2.5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade.cleaning import clean
from atrade.config import DEFAULTS
from atrade.data import is_intraday, parse_price_csv, to_weekly
from atrade.demo import demo_prices

S = dict(DEFAULTS)


def test_weekly_bars_combine_each_week():
    idx = pd.bdate_range("2024-01-01", periods=8)            # Mon 1 Jan .. Wed 10 Jan
    d = pd.DataFrame({"Open": np.arange(8) + 10.0, "High": np.arange(8) + 12.0, "Low": np.arange(8) + 9.0,
                      "Close": np.arange(8) + 11.0, "Volume": np.ones(8)}, index=idx)
    w = to_weekly(d)
    assert list(w.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-10")]   # dated by the week's last day
    assert w.iloc[0].tolist() == [10.0, 16.0, 9.0, 15.0, 5.0]
    assert w.iloc[1].tolist() == [15.0, 19.0, 14.0, 18.0, 3.0]


def test_csv_files_in_several_layouts():
    a = b"date,open,high,low,close,volume\n2024-01-02,1,2,0.5,1.5,100\n2024-01-03,1.5,2.5,1,2,200\n"
    df = parse_price_csv(a)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"] and len(df) == 2
    b = b"Date,AAPL.Open,AAPL.High,AAPL.Low,AAPL.Close\n2024-01-02 09:30:00-05:00,1,2,0.5,1.5\n"
    df = parse_price_csv(b)
    assert df["Volume"].iloc[0] == 0 and df.index[0] == pd.Timestamp("2024-01-02 09:30")   # local time kept
    assert is_intraday(df.index) and not is_intraday(pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="Date"):
        parse_price_csv(b"Day,Open,High,Low,Close\n1,1,1,1,1\n")
    with pytest.raises(ValueError, match="Close"):
        parse_price_csv(b"Date,Open,High,Low\n2024-01-02,1,1,1\n")


def test_cleaning_keeps_hourly_bars():
    days = pd.bdate_range("2024-01-02", periods=5)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=9.5 + k) for d in days for k in range(7)])
    c = 100 + np.arange(len(idx)) * 0.1
    raw = pd.DataFrame({"Open": c, "High": c + 0.2, "Low": c - 0.2, "Close": c, "Volume": 1.0}, index=idx)
    df, _, q = clean(raw, S)
    assert len(df) == len(raw) and df.index.equals(idx) and q["passed"]


def test_weekly_analysis_switches_off_daily_options_rules():
    from atrade.analysis import analyze
    r = analyze("DEMO", S, prices=demo_prices(), bar="week")
    assert r["bar"] == "week" and (pd.Series(r["df"].index).diff().dropna().dt.days >= 1).all()
    assert len(r["df"]) < 0.25 * len(demo_prices())
    assert not [x for x in r["signals"] if x.group == "sentiment"]


def test_weekly_bars_no_look_ahead_for_complete_weeks():
    """Pattern signals on complete weeks are the same whether or not later weeks exist."""
    from atrade.patterns import all_patterns
    df, _, _ = clean(demo_prices(), S)
    w = to_weekly(df)
    cut = 600
    def sigs(x):
        c = x["Close"].to_numpy(float)
        return sorted((g.rule, g.idx, g.direction) for g in all_patterns(c, S, np.zeros(len(c)), x["High"].to_numpy(float),
                                                                      x["Low"].to_numpy(float)) if g.idx <= cut)
    part = to_weekly(df[df.index <= w.index[cut]])
    assert part.index[-1] == w.index[cut] and sigs(w) == sigs(part)


def test_dates_after_the_last_bar_follow_the_bar_size():
    from atrade.data import bar_date
    days = pd.bdate_range("2024-01-01", periods=5)                 # ends Fri 5 Jan
    assert bar_date(days, 6) == pd.Timestamp("2024-01-09")         # 2 trading days later
    weeks = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])
    assert bar_date(weeks, 11, "week") == pd.Timestamp("2024-03-22")   # 10 Fridays later
    assert bar_date(pd.DatetimeIndex(["2024-01-05", "2024-01-09"]), 2, "week") == pd.Timestamp("2024-01-19")  # last week still building
