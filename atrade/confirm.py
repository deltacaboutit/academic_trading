"""Textbook confirmation checks for a chart pattern (ported from PatternFinder 2.5).

Classic chart books say a pattern alone is not enough: volume, the trend and the
way the breakout happens should agree with it ("the principle of confirmation",
Murphy). This module turns those rules into 21 yes/no checks. The keys,
thresholds and sources are the same as in PatternFinder 2.5, so the results of
its v2.5 back-test apply here too. Every threshold was fixed in advance from the
sources below, NOT tuned on back-test results.

Sources
  TA101   StockCharts ChartSchool, "Technical Analysis 101", parts 4 to 16
  Murphy  John J. Murphy, "Charting Made Easy" (ChartSchool e-book)
  E&M     Edwards & Magee, "Technical Analysis of Stock Trends"
  Bulk.   Bulkowski, "Encyclopedia of Chart Patterns"

Each check answers one question and gives:
  applies  does the check make sense for this pattern?
  passed   True or False, or None when it cannot be worked out (not enough
           history, no breakout yet, no market data). None is not a failure.
  value    the number behind the answer
  text     the answer in plain words

The checks are information only: they never change a signal or a vote. They use
only prices up to the day they are judged on (no look-ahead).

Differences from PatternFinder 2.5 (tool choices, so the checks fit this tool):
  * ATR is Wilder's 14-day Average True Range from daily highs and lows, as in
    PatternFinder 2.5.
  * "Room to run" looks for old peaks or troughs among this tool's rolling-window
    turning points with a 15-day window (the book's window for double and triple
    tops), counting only points already confirmed on the judging day.
    PatternFinder 2.5 used its 3-ATR swing points instead.
  * Wedges count as continuation patterns, because the book's wedge rule used here
    needs a move in the breakout's direction before the wedge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .pattern_info import PATTERN_INFO
from .signals import Setup, Signal
from .turning_points import rolling_window

TOPS_BOTTOMS = {"Double top", "Double bottom", "Triple top", "Triple bottom",
                "Head-and-shoulders top", "Inverse head-and-shoulders"}
REVERSAL = {k for k, v in PATTERN_INFO.items() if v["category"] == "Reversal"}
CONTINUATION = {k for k, v in PATTERN_INFO.items() if v["category"] in ("Continuation", "Either way")}
TRIANGLES = {"Ascending triangle", "Descending triangle", "Symmetrical triangle"}

# ---- Fixed thresholds (decided in advance from the sources) ----------------
HEAVY_VOLUME = 1.5          # breakout volume at least 1.5x its 20-day average (Bulk.; E&M "heavy volume")
PRIOR_TREND_DAYS = 60       # about 3 months: Murphy's "secondary trend" is 1 to 3 months
PRIOR_TREND_ATRS = 5.0      # a trend to reverse = a move of at least 5 ATRs over those 3 months (PatternFinder 2.5's choice)
MAX_RETRACE = 2 / 3         # Murphy: a pullback of more than two-thirds warns the trend is reversing
CHASE_ATRS = 1.0            # breakout close no more than 1 ATR beyond the line (PatternFinder 2.5's choice)
TRIANGLE_WINDOW = (0.5, 0.75)  # Murphy: symmetrical triangles break out halfway to three-quarters of the way
RS_DAYS = 126               # relative strength over about 6 months (momentum research uses 3 to 12 months)
ROOM_LOOKBACK = 250         # look for old peaks and heavy-trading zones in the year before the pattern
ROOM_W = 15                 # tool choice: turning-point window for "old peaks" (book's double/triple-top window)
KR_LOOKBACK = 10            # key reversal day: a new low (high) for at least the last 10 days


@dataclass
class CheckDef:
    key: str
    group: str
    label: str      # short name shown in the app
    question: str   # what it asks, in plain words
    source: str


CHECKS = [
    # ---- Volume ----
    CheckDef("volume_trend", "Volume", "Volume favours the breakout direction",
             "In the second half of the pattern, was volume heavier on days that moved in the breakout's "
             "direction than on days that moved against it?", "TA101 part 10; Murphy"),
    CheckDef("volume_dryup", "Volume", "Volume dried up during the pattern",
             "Was average volume in the last third of the pattern lower than in the first third?",
             "TA101 part 12; Murphy"),
    CheckDef("breakout_volume", "Volume", "Heavy volume on the breakout day",
             f"Was breakout-day volume at least {HEAVY_VOLUME}x its 20-day average?", "TA101 parts 12, 15; Murphy"),
    CheckDef("obv_confirms", "Volume", "On-Balance Volume confirms",
             "At the breakout, was OBV at or beyond its level at the pattern's own extreme "
             "(no divergence between price and volume)?", "Murphy"),
    CheckDef("obv_leads", "Volume", "On-Balance Volume broke out first",
             "Did OBV already make a new high (or low) for the pattern in the 10 days before the price did?",
             "Murphy"),
    CheckDef("last_peak_lighter", "Volume", "Lighter volume on the last peak or trough",
             "For tops and bottoms: was volume around the last peak (or trough) lower than around the first?",
             "TA101 parts 10, 13; E&M; Bulk."),
    # ---- Trend ----
    CheckDef("prior_trend", "Trend", "There was a trend to reverse",
             f"For reversal patterns: did the price move at least {PRIOR_TREND_ATRS:.0f} ATRs the other way in the "
             f"{PRIOR_TREND_DAYS} days before the pattern?", "TA101 parts 8, 13; Murphy"),
    CheckDef("with_trend", "Trend", "Breaks out in the direction of the prior trend",
             f"For continuation patterns: does the breakout go the same way as the {PRIOR_TREND_DAYS} days "
             "before the pattern?", "TA101 parts 11, 12; Murphy"),
    CheckDef("major_trend", "Trend", "Agrees with the 200-day trend",
             "Bullish: close above a rising 200-day average. Bearish: below a falling one.", "Murphy"),
    CheckDef("weekly_trend", "Trend", "Agrees with the weekly chart",
             "Bullish: 10-week average above the 40-week average. Bearish: below.", "Murphy"),
    CheckDef("market_trend", "Trend", "Agrees with the market (SPY)",
             "Bullish: SPY above its 200-day average. Bearish: below.", "Murphy (top-down approach)"),
    CheckDef("relative_strength", "Trend", "Relative strength agrees",
             f"Bullish: the stock beat SPY over the last {RS_DAYS} days (about 6 months). Bearish: it lagged.",
             "Murphy; Jegadeesh & Titman (1993)"),
    # ---- Retracement ----
    CheckDef("pullback_depth", "Retracement", "Pullback not deeper than two-thirds",
             "For continuation patterns: did the pattern give back no more than two-thirds of the move "
             "before it?", "Murphy"),
    # ---- Breakout quality ----
    CheckDef("strong_close", "Breakout", "Strong close on the breakout day",
             "Did the breakout day close in the top third of its range (bottom third for bearish)?",
             "TA101 parts 4, 5"),
    CheckDef("not_chasing", "Breakout", "Close not too far past the line",
             f"Was the breakout close within {CHASE_ATRS:.0f} ATR of the breakout line?", "PatternFinder 2.5's choice"),
    CheckDef("rsi_ok", "Breakout", "Not overbought / oversold",
             "Bullish: RSI(14) at or below 70. Bearish: at or above 30.", "Murphy"),
    CheckDef("inside_bands", "Breakout", "Inside the Bollinger Bands",
             "Did the breakout close stay inside the Bollinger Bands (20 days, 2 standard deviations)?", "Murphy"),
    CheckDef("triangle_timing", "Breakout", "Triangle broke out at the right time",
             "For triangles: did the breakout come between halfway and three-quarters of the way to the tip?",
             "Murphy; Bulk."),
    # ---- Gaps and key reversal days ----
    CheckDef("breakaway_gap", "Gaps & reversal days", "Breakaway gap on heavy volume",
             "Did the breakout day gap past the previous day's range, with above-average volume?",
             "TA101 part 15; Murphy"),
    CheckDef("key_reversal", "Gaps & reversal days", "Key reversal day at the turn",
             "For reversal patterns and cup-and-handle: was there a key reversal day within 2 days of the "
             "pattern's last turning point?", "Murphy"),
    # ---- Support and resistance ----
    CheckDef("room_to_run", "Room to run", "Room to run to the target",
             f"Is the path to the target free of old major peaks (or troughs) and of the busiest trading zone "
             f"(volume-by-price) from the {ROOM_LOOKBACK} days before the pattern?", "TA101 part 7"),
]
CHECK_KEYS = [c.key for c in CHECKS]
CHECK_BY_KEY = {c.key: c for c in CHECKS}

# Checks that need a breakout day (left blank for patterns still forming)
NEEDS_BREAKOUT = {"breakout_volume", "obv_confirms", "obv_leads", "strong_close", "not_chasing", "rsi_ok",
                  "inside_bands", "triangle_timing", "breakaway_gap", "room_to_run"}

# Checks a back-test has shown to matter (Holm p < 0.05, better in both halves of the period,
# and at least 20 trades that passed and 20 that failed). Empty until those results are in:
# the app then shows every check but says none is proven yet.
PROVEN: set = set()


def key_reversal_days(o, h, l, c, v):
    """Arrays marking key reversal days (Murphy).

    Bullish key reversal: a new low for the last 10 days, an open below the previous
    close, and a close above the previous close ("opens lower and closes higher").
    Bearish is the mirror image. It is "strong" when it is also an outside day (high
    and low both beyond the previous day's range) on at least 1.5x average volume.
    Returns (bull, bear, strong_bull, strong_bear) as boolean arrays.
    """
    o, h, l, c, v = (np.asarray(x, float) for x in (o, h, l, c, v))
    n = len(c)
    if n < KR_LOOKBACK + 1:
        z = np.zeros(n, bool)
        return z, z.copy(), z.copy(), z.copy()
    low10 = pd.Series(l).rolling(KR_LOOKBACK, min_periods=KR_LOOKBACK).min().to_numpy()
    high10 = pd.Series(h).rolling(KR_LOOKBACK, min_periods=KR_LOOKBACK).max().to_numpy()
    pc, ph, pl = np.r_[np.nan, c[:-1]], np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    with np.errstate(invalid="ignore"):
        bull = (l <= low10) & (l < pl) & (o < pc) & (c > pc)
        bear = (h >= high10) & (h > ph) & (o > pc) & (c < pc)
        avg_v = pd.Series(v).rolling(20, min_periods=5).mean().shift(1).to_numpy()
        outside = (h > ph) & (l < pl)
        heavy = v >= HEAVY_VOLUME * avg_v
    bull = np.nan_to_num(bull).astype(bool)
    bear = np.nan_to_num(bear).astype(bool)
    return bull, bear, bull & outside & heavy, bear & outside & heavy


class Context:
    """Everything the checks need, worked out once per price table.

    Every array value on day t uses only prices up to day t, so judging a check on
    day i never looks past day i.
    """

    def __init__(self, df: pd.DataFrame, market: pd.Series | None = None):
        self.df = df
        self.o, self.h, self.l, self.c, self.v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        self.n = len(df)
        self.atr = np.maximum(ind.atr_filled(self.h, self.l, self.c), 1e-9)
        self.obv = ind.obv(self.c, self.v)
        self.rsi = ind.rsi(self.c, 14)
        bb = ind.bollinger(self.c, 20, 2.0)
        self.bb_up, self.bb_lo = bb["up_usual"], bb["low_usual"]      # the usual Bollinger Bands, as in PatternFinder 2.5
        self.sma200 = ind.sma(self.c, 200)
        self.vavg20 = ind.shift(ind.sma(self.v, 20))                  # average of the 20 days BEFORE each day
        self.kr_bull, self.kr_bear, self.kr_sbull, self.kr_sbear = key_reversal_days(self.o, self.h, self.l, self.c, self.v)
        self.weekly = None
        if isinstance(df.index, pd.DatetimeIndex) and self.n > 1:
            self.week_end = df.index.to_period("W-FRI").to_timestamp(how="end").normalize()
            self.weekly = pd.Series(self.c, index=df.index).groupby(self.week_end).last()
        self.mkt = self.mkt_sma200 = None
        if market is not None and len(market):
            m = pd.Series(market).astype(float)
            m.index = pd.DatetimeIndex(m.index)
            m = m[~m.index.duplicated(keep="last")].sort_index()
            both = m.index.union(df.index)
            self.mkt = m.reindex(both).ffill().reindex(df.index).to_numpy(float)
            self.mkt_sma200 = (pd.Series(ind.sma(m.to_numpy(float), 200), index=m.index)
                               .reindex(both).ffill().reindex(df.index).to_numpy(float))
        tp = rolling_window(self.c, ROOM_W)
        self.tp = (tp["idx"].to_numpy(int), tp["price"].to_numpy(float), tp["kind"].to_numpy(str),
                   tp["known"].to_numpy(int))

    def weekly_trend(self, i):
        """+1 if the 10-week average is above the 40-week average using weeks up to day i, -1 if below."""
        if self.weekly is None:
            return None
        this_week = self.week_end[i]
        past = self.weekly[self.weekly.index < this_week]
        closes = np.r_[past.to_numpy(float), self.c[i]]     # this week so far = today's close
        if len(closes) < 40:
            return None
        return float(np.sign(closes[-10:].mean() - closes[-40:].mean()))

    def old_extremes(self, i, kind, a, b):
        """Prices of turning points of `kind` ('P' or 'T') on days a..b-1 that are confirmed by day i."""
        idx, pr, kd, kn = self.tp
        sel = (idx >= a) & (idx < b) & (kd == kind) & (kn <= i)
        return pr[sel]


def _avg(x):
    x = np.asarray(x, float)
    return float(x.mean()) if x.size else float("nan")


def _turn_index(points, d):
    """The pattern's last turning point in the direction it is turning FROM
    (the last trough for a bullish pattern, the last peak for a bearish one)."""
    pts = sorted(points, key=lambda x: x[0])
    for k in range(len(pts) - 1, 0, -1):
        if (d > 0 and pts[k][1] < pts[k - 1][1]) or (d < 0 and pts[k][1] > pts[k - 1][1]):
            return int(pts[k][0])
    return int(pts[-1][0]) if pts else None


def _shape(item):
    """(rule, direction, judging day, broke out?, line level, target, points, apex) for a Signal or a Setup."""
    pts = [p for p in item.points if p[2] != "prior"]
    if isinstance(item, Signal):
        lv = item.line_value(item.idx)
        return (item.rule, item.direction, item.idx, True, lv if lv is not None else float("nan"),
                item.target if item.target is not None else float("nan"), pts, item.extra.get("apex"))
    return item.rule, item.direction, None, False, float("nan"), float("nan"), pts, None


def run_checks(ctx: Context, item: Signal | Setup, i: int | None = None) -> dict:
    """Work out every check for a pattern, judged on day i.

    For a Signal, i is its breakout day. For a Setup (no breakout yet), i is the
    last day (or the day it failed or expired); the breakout-day checks are left blank.
    Returns {key: {"applies", "passed", "value", "text"}}.
    """
    name, d, i_sig, broke, level, target, pts, apex = _shape(item)
    if i is None:
        i = i_sig if broke else min(ctx.n - 1, item.resolved if item.resolved is not None else ctx.n - 1)
    o, h, l, c, v = ctx.o, ctx.h, ctx.l, ctx.c, ctx.v
    out = {}

    def put(key, applies, passed=None, value=float("nan"), text=""):
        out[key] = {"applies": bool(applies), "passed": None if passed is None else bool(passed),
                    "value": float(value) if value is not None else float("nan"), "text": text}

    if d == 0 or i is None or i < 1 or not pts:
        for k in CHECK_KEYS:
            put(k, False, text="the pattern can still break either way")
        return out
    s = int(pts[0][0])
    is_rev, is_cont = name in REVERSAL, name in CONTINUATION
    a = ctx.atr[i]
    end = i if broke else i + 1    # pattern days: s .. end-1 (not the breakout day itself)

    # ---------------- Volume ----------------
    mid = s + (end - s) // 2
    days = np.arange(max(mid, 1), end)
    move = np.sign(c[days] - c[days - 1])
    with_v, against_v = v[days][move == d], v[days][move == -d]
    if len(with_v) >= 3 and len(against_v) >= 3 and against_v.mean() > 0:
        r = with_v.mean() / against_v.mean()
        put("volume_trend", True, r > 1.0, r, f"volume on days moving the breakout's way was {r:.2f}x the other days")
    else:
        put("volume_trend", True, None, text="too few days to judge")

    span = end - s
    if span >= 9:
        third = span // 3
        first, last = _avg(v[s:s + third]), _avg(v[end - third:end])
        r = last / first if first > 0 else float("nan")
        put("volume_dryup", True, r < 1.0 if np.isfinite(r) else None, r,
            f"volume in the last third was {r:.2f}x the first third")
    else:
        put("volume_dryup", True, None, text="pattern too short to judge")

    if broke:
        avg = ctx.vavg20[i]
        r = v[i] / avg if np.isfinite(avg) and avg > 0 else float("nan")
        put("breakout_volume", True, (r >= HEAVY_VOLUME) if np.isfinite(r) else None, r,
            f"breakout volume was {r:.1f}x its 20-day average")
        seg = np.arange(s, i)
        if seg.size:
            ext_i = seg[np.argmax(h[seg])] if d > 0 else seg[np.argmin(l[seg])]
            ok = ctx.obv[i] >= ctx.obv[ext_i] if d > 0 else ctx.obv[i] <= ctx.obv[ext_i]
            put("obv_confirms", True, ok, ctx.obv[i] - ctx.obv[ext_i],
                "OBV " + ("confirmed the breakout" if ok else "lagged behind the price (a divergence)"))
        else:
            put("obv_confirms", True, None)
        if i - 10 > s + 5:
            early, late = ctx.obv[s:i - 10], ctx.obv[i - 10:i]
            ok = late.max() > early.max() if d > 0 else late.min() < early.min()
            put("obv_leads", True, ok, float("nan"),
                ("OBV made a new " + ("high" if d > 0 else "low") + " before the price") if ok
                else "OBV did not lead the price")
        else:
            put("obv_leads", True, None, text="pattern too short to judge")
    else:
        for k in ("breakout_volume", "obv_confirms", "obv_leads"):
            put(k, True, None, text="worked out at the breakout")

    if name in TOPS_BOTTOMS:
        ex = sorted(pts, key=lambda x: x[0])[::2]           # the peaks (or troughs), not the dips between them
        if len(ex) >= 2:
            v1 = _avg(v[max(0, ex[0][0] - 2):ex[0][0] + 3])
            v2 = _avg(v[max(0, ex[-1][0] - 2):min(ex[-1][0] + 3, i + 1)])
            r = v2 / v1 if v1 > 0 else float("nan")
            put("last_peak_lighter", True, r < 1.0 if np.isfinite(r) else None, r,
                f"volume at the last turn was {r:.2f}x the first")
        else:
            put("last_peak_lighter", True, None)
    else:
        put("last_peak_lighter", False)

    # ---------------- Trend ----------------
    if s - PRIOR_TREND_DAYS >= 0:
        mv = c[s] - c[s - PRIOR_TREND_DAYS]
        atrs = mv / max(ctx.atr[s], 1e-9)
        if is_rev:
            put("prior_trend", True, -d * atrs >= PRIOR_TREND_ATRS, -d * atrs,
                f"the price moved {abs(atrs):.1f} ATRs {'up' if mv > 0 else 'down'} in the {PRIOR_TREND_DAYS} days before")
        else:
            put("prior_trend", False)
        if is_cont:
            put("with_trend", True, d * mv > 0, 100 * mv / c[s - PRIOR_TREND_DAYS],
                f"the {PRIOR_TREND_DAYS} days before moved {100 * mv / c[s - PRIOR_TREND_DAYS]:+.1f}%")
        else:
            put("with_trend", False)
    else:
        put("prior_trend", is_rev, None, text="not enough history")
        put("with_trend", is_cont, None, text="not enough history")

    if i >= 220 and np.isfinite(ctx.sma200[i]) and np.isfinite(ctx.sma200[i - 20]):
        rising = ctx.sma200[i] > ctx.sma200[i - 20]
        ok = (c[i] > ctx.sma200[i] and rising) if d > 0 else (c[i] < ctx.sma200[i] and not rising)
        put("major_trend", True, ok, 100 * (c[i] / ctx.sma200[i] - 1),
            f"close {100 * (c[i] / ctx.sma200[i] - 1):+.1f}% from a {'rising' if rising else 'falling'} 200-day average")
    else:
        put("major_trend", True, None, text="needs 220 days of history")

    wt = ctx.weekly_trend(i)
    put("weekly_trend", True, None if wt is None else wt == d, wt if wt is not None else float("nan"),
        "" if wt is None else f"10-week average is {'above' if wt > 0 else 'below'} the 40-week average")

    if ctx.mkt is not None and np.isfinite(ctx.mkt[i]) and np.isfinite(ctx.mkt_sma200[i]):
        up = ctx.mkt[i] > ctx.mkt_sma200[i]
        put("market_trend", True, up == (d > 0), 100 * (ctx.mkt[i] / ctx.mkt_sma200[i] - 1),
            f"SPY is {'above' if up else 'below'} its 200-day average")
    else:
        put("market_trend", True, None, text="no market data")

    if ctx.mkt is not None and i - RS_DAYS >= 0 and np.isfinite(ctx.mkt[i - RS_DAYS]) and ctx.mkt[i - RS_DAYS] > 0:
        rs = 100 * ((c[i] / c[i - RS_DAYS]) - (ctx.mkt[i] / ctx.mkt[i - RS_DAYS]))
        put("relative_strength", True, d * rs > 0, rs, f"the stock {'beat' if rs > 0 else 'lagged'} SPY by "
            f"{abs(rs):.1f} points over {RS_DAYS} days")
    else:
        put("relative_strength", True, None, text="no market data")

    # ---------------- Retracement ----------------
    if is_cont and s - PRIOR_TREND_DAYS >= 0 and end - s >= 2:
        seg = np.arange(s, end)
        if d > 0:
            base = l[s - PRIOR_TREND_DAYS:s + 1].min()
            top_i = seg[np.argmax(h[seg])]
            back = h[top_i] - l[top_i:end].min()
            size = h[top_i] - base
        else:
            base = h[s - PRIOR_TREND_DAYS:s + 1].max()
            top_i = seg[np.argmin(l[seg])]
            back = h[top_i:end].max() - l[top_i]
            size = base - l[top_i]
        if size > 0:
            r = back / size
            put("pullback_depth", True, r <= MAX_RETRACE, 100 * r, f"the pattern gave back {100 * r:.0f}% of the move before it")
        else:
            put("pullback_depth", True, None, text="no clear move before the pattern")
    else:
        put("pullback_depth", is_cont, None, text="not enough history")

    # ---------------- Breakout quality ----------------
    if broke:
        rng_ = h[i] - l[i]
        if rng_ > 0:
            pos = (c[i] - l[i]) / rng_
            put("strong_close", True, pos >= 2 / 3 if d > 0 else pos <= 1 / 3, 100 * pos,
                f"closed at {100 * pos:.0f}% of the day's range")
        else:
            put("strong_close", True, None)
        if np.isfinite(level):
            dist = d * (c[i] - level) / a
            put("not_chasing", True, dist <= CHASE_ATRS, dist, f"closed {dist:.1f} ATRs past the line")
        else:
            put("not_chasing", True, None)
        r = ctx.rsi[i]
        put("rsi_ok", True, None if not np.isfinite(r) else (r <= 70 if d > 0 else r >= 30), r, f"RSI was {r:.0f}")
        if np.isfinite(ctx.bb_up[i]):
            ok = c[i] <= ctx.bb_up[i] if d > 0 else c[i] >= ctx.bb_lo[i]
            put("inside_bands", True, ok, float("nan"), "inside the bands" if ok else "outside the bands")
        else:
            put("inside_bands", True, None)
        if name in TRIANGLES:
            if apex is not None and apex > s:
                pos = (i - s) / (apex - s)
                lo_, hi_ = TRIANGLE_WINDOW
                put("triangle_timing", True, lo_ <= pos <= hi_, 100 * pos, f"broke out {100 * pos:.0f}% of the way to the tip")
            else:
                put("triangle_timing", True, None)
        else:
            put("triangle_timing", False)
        gap = (l[i] > h[i - 1]) if d > 0 else (h[i] < l[i - 1])
        avg = ctx.vavg20[i]
        heavy = np.isfinite(avg) and v[i] > avg
        gsize = 100 * ((l[i] / h[i - 1] - 1) if d > 0 else (1 - h[i] / l[i - 1]))
        put("breakaway_gap", True, bool(gap and heavy), gsize if gap else 0.0,
            "gapped past the previous day on above-average volume" if gap and heavy
            else ("gapped, but on light volume" if gap else "no gap"))
    else:
        for k in ("strong_close", "not_chasing", "rsi_ok", "inside_bands", "breakaway_gap"):
            put(k, True, None, text="worked out at the breakout")
        put("triangle_timing", name in TRIANGLES, None, text="worked out at the breakout")

    if is_rev or name == "Cup and handle":
        t_i = _turn_index(pts, d)
        if t_i is not None:
            js = np.arange(max(1, t_i - 2), min(i, t_i + 2) + 1)
            arr, strong = (ctx.kr_bull, ctx.kr_sbull) if d > 0 else (ctx.kr_bear, ctx.kr_sbear)
            hit = js[arr[js]] if js.size else js
            if hit.size:
                st = bool(strong[hit].any())
                put("key_reversal", True, True, 2.0 if st else 1.0,
                    "a " + ("strong (outside day, heavy volume) " if st else "") + "key reversal day at the turn")
            else:
                put("key_reversal", True, False, 0.0, "no key reversal day at the turn")
        else:
            put("key_reversal", True, None)
    else:
        put("key_reversal", False)

    # ---------------- Room to run ----------------
    if broke and np.isfinite(target) and np.isfinite(level):
        lo_p, hi_p = sorted((level + d * 0.5 * a, target))
        blockers = []
        old = ctx.old_extremes(i, "P" if d > 0 else "T", s - ROOM_LOOKBACK, s)
        inside = old[(old > lo_p) & (old < hi_p)]
        if inside.size:
            blockers.append(f"an old {'peak' if d > 0 else 'trough'} at {inside[-1]:,.2f}")
        a0 = max(0, s - ROOM_LOOKBACK)
        if s - a0 >= 40:
            closes, vols = c[a0:s], v[a0:s]
            edges = np.linspace(l[a0:s].min(), h[a0:s].max(), 21)
            vol_bins = np.histogram(closes, bins=edges, weights=vols)[0]
            centers = (edges[:-1] + edges[1:]) / 2
            busiest = centers[int(np.argmax(vol_bins))]   # the "point of control" of volume-by-price
            if lo_p < busiest < hi_p:
                blockers.append(f"the year's busiest trading zone, around {busiest:,.2f}")
        put("room_to_run", True, not blockers, float(len(blockers)),
            "nothing in the way" if not blockers else "in the way: " + " and ".join(blockers))
    else:
        put("room_to_run", True, None, text="worked out at the breakout")
    return out


def summarize(checks: dict) -> tuple[int, int]:
    """(passed, judged): how many checks passed out of those that could be judged."""
    judged = [x for x in checks.values() if x["applies"] and x["passed"] is not None]
    return sum(x["passed"] for x in judged), len(judged)


def attach_checks(df: pd.DataFrame, items: list, market: pd.Series | None = None) -> None:
    """Work out the checks for every pattern Signal or Setup and store them in item.extra["checks"]."""
    if not items:
        return
    ctx = Context(df, market)
    for it in items:
        chk = run_checks(ctx, it)
        it.extra["checks"] = chk
        it.extra["checks_passed"], it.extra["checks_judged"] = summarize(chk)


def flat_columns(checks: dict) -> dict:
    """The checks as flat columns: chk_<key> = 1 / 0 / blank, val_<key> = number (same as PatternFinder 2.5)."""
    row = {}
    for k in CHECK_KEYS:
        x = checks.get(k, {"applies": False, "passed": None, "value": float("nan")})
        row[f"chk_{k}"] = (1 if x["passed"] else 0) if (x["applies"] and x["passed"] is not None) else np.nan
        row[f"val_{k}"] = round(x["value"], 4) if np.isfinite(x["value"]) else np.nan
    passed, judged = summarize(checks)
    row["checks_passed"], row["checks_judged"] = passed, judged
    return row
