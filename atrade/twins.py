"""The random-twin test for chart patterns (ported from PatternFinder 2.5's backtest.py).

The question: do the PATTERNS add anything, or would any trade of the same
shape, on the same stock at about the same time, have done as well?

A fair comparison ("random twins")
  Every real pattern trade is compared with 200 pretend trades on the same stock,
  in the same direction, with the same target and stop distances (in %), the same
  broken line (as a % of the entry price, day by day) and the same time limit,
  but started on random days within about a year (250 trading days) either side
  of the real trade, so both face similar market conditions.
  Edge = the real trade's return - the average return of its twins.
  If the patterns are useful, the edge should be clearly above zero.

Realistic fills (PatternFinder 2.5)
  * Entry at the next day's open, as always. If that open is already past the
    stop or the target, the trade is skipped.
  * The target or stop is hit when the day's HIGH or LOW reaches it (not just the
    close). If the price gaps through a level overnight, the trade exits at the
    open (the realistic, less flattering price).
  * If the stop and the target are both touched on the same day, the stop counts.
  * Costs: the tool's commission plus half-spread, taken at entry and at exit.

Four ways of managing the same trade, all compared on the SAME twin days
  book          the tool's own plan (Framework 8): the pattern's target and time
                limit; out after 2 closes back across the broken line, or at a
                4% loss.
  kr_exit       the book plan, plus: get out at the next open after a STRONG key
                reversal day against the trade (an outside day on heavy volume;
                Murphy). From PatternFinder 2.5.
  gap_stop      for breakouts that came with a breakaway gap: the book plan, but
                with the stop just beyond the far side of the gap (a quarter of an
                ATR past yesterday's high or low; Murphy: if the gap fills, the
                breakout has failed). Only used when it is tighter than the 4%
                stop. From PatternFinder 2.5.
  pattern_stop  the pattern's own stop from the chart books, as in PatternFinder
                2.5 (for example just above the right shoulder of a head-and-
                shoulders top, a quarter of an ATR beyond it), with no 4% stop and
                no line stop. This is the "pattern-specific stops" variant; the
                book plan stays the tool's default.

No look-ahead: the pattern signals are the same ones the rating uses, which only
use prices up to their signal day (tests/test_core.py::test_no_look_ahead), and the
trade starts at the next day's open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .confirm import _turn_index, flat_columns, key_reversal_days

N_RANDOM = 200        # random twins per real trade (PatternFinder 2.5)
TWIN_WINDOW = 250     # twins start within this many trading days (about a year) of the real trade
FIRST_DAY = 250       # twins never start in the first year of data (PatternFinder 2.5's warm-up)
STOP_BUFFER = 0.25    # pattern stops sit a quarter of an ATR beyond the level (PatternFinder 2.5)
VARIANTS = {
    "book": "The tool's own plan: target, time limit, 2 closes back across the line or a 4% loss",
    "kr_exit": "Book plan, plus exit at the next open after a strong key reversal day against the trade",
    "gap_stop": "Book plan with the stop beyond the breakaway gap (gap breakouts only, when tighter than 4%)",
    "pattern_stop": "The pattern's own stop from the chart books instead of the 4% and line stops",
}


def simulate(o, h, l, c, t, d, target, stop, hold, line_rel=None, entry=None, stop_days=2,
             exit_flags=None, cost=0.001):
    """One trade signalled on day t (known at its close), entered at the open of day t + 1.

    target, stop  price levels (stop=None: no fixed stop)
    hold          the time limit, in days after the signal day
    line_rel      the broken line as a fraction of the entry price on days t, t+1, ...
                  (None: no line stop). The trade closes after `stop_days` closes in a
                  row back across it, at that close.
    exit_flags    a True/False value per day; True at a close while the trade is open
                  means exit at the next day's open (the key-reversal variant).
    cost          per side (commission plus half-spread).
    Returns (return after costs, exit day, reason) or None if the trade cannot be taken.
    """
    n = len(c)
    if t + 1 >= n:
        return None
    e = o[t + 1] if entry is None else entry
    if not np.isfinite(e) or e <= 0:
        return None
    if d > 0 and not ((stop is None or stop < e) and e < target):
        return None
    if d < 0 and not ((stop is None or stop > e) and e > target):
        return None
    last = min(n - 1, t + max(1, int(hold)))
    px, why, j = c[last], "time", last
    back = 0
    for j in range(t + 1, last + 1):
        gap_ok = j > t + 1                       # on the entry day the open IS the entry price
        if d > 0:
            if stop is not None and gap_ok and o[j] <= stop:
                px, why = o[j], "stop"; break
            if stop is not None and l[j] <= stop:
                px, why = stop, "stop"; break
            if gap_ok and o[j] >= target:
                px, why = o[j], "target"; break
            if h[j] >= target:
                px, why = target, "target"; break
        else:
            if stop is not None and gap_ok and o[j] >= stop:
                px, why = o[j], "stop"; break
            if stop is not None and h[j] >= stop:
                px, why = stop, "stop"; break
            if gap_ok and o[j] <= target:
                px, why = o[j], "target"; break
            if l[j] <= target:
                px, why = target, "target"; break
        if line_rel is not None and j - t < len(line_rel):
            lv = e * line_rel[j - t]
            back = back + 1 if ((d > 0 and c[j] < lv) or (d < 0 and c[j] > lv)) else 0
            if back >= stop_days:
                px, why = c[j], "line"; break
        if exit_flags is not None and j < last and exit_flags[j]:
            j += 1
            px, why = o[j], "reversal day"; break
    else:
        j = last
        if last < t + hold:
            why = "end of data"
    return d * (px - e) / e - 2 * cost, j, why


def pattern_stop(sg, h, l, atr) -> float | None:
    """The pattern's own stop-loss price, as the chart books (and PatternFinder 2.5) place it.

    Bullish patterns: a quarter of an ATR below the lowest LOW of the relevant part
    of the pattern; bearish: the mirror image. Rectangles: the middle of the box.
    """
    pts = sorted([p for p in sg.points if p[2] != "prior"], key=lambda x: x[0])
    e, d, r = int(sg.idx), sg.direction, sg.rule
    if not pts:
        return None
    buf = STOP_BUFFER * atr[e]
    if r == "Rectangle":
        lo = (sg.extra.get("height") or 0.0)
        lv = sg.line_value(e)
        return None if lv is None else lv - d * 0.5 * lo
    if r in ("Head-and-shoulders top", "Inverse head-and-shoulders"):
        days = [pts[-1][0]]                                # the right shoulder
    elif r.startswith(("Double", "Triple")):
        days = [p[0] for p in pts[::2]]                    # the peaks (or troughs)
    elif r.startswith(("Bull flag", "Bear flag", "Bull pennant", "Bear pennant")):
        days = list(range(pts[0][0], e))                   # the whole flag or pennant
    elif r == "Cup and handle":
        hd = [p[0] for p in pts if p[2] == "handle"]
        days = hd or [pts[-1][0]]                          # the handle's low
    elif r.startswith("Rounding"):
        a, b = pts[0][0], pts[-1][0]
        days = list(range(a + 2 * (b - a) // 3, b))        # the last third of the bowl or dome
    elif r.startswith("Island"):
        days = list(range(pts[0][0] + 1, pts[-1][0]))      # the island days
    else:                                                  # wedges and triangles: the most recent swing
        k = _turn_index(pts, d)
        days = [k] if k is not None else [pts[-1][0]]
    days = [x for x in days if 0 <= x <= e]
    if not days:
        return None
    return float(np.min(l[days]) - buf) if d > 0 else float(np.max(h[days]) + buf)


def _line_rel(sg, entry, hold):
    if sg.line is None:
        return None
    return np.array([sg.line_value(sg.idx + k) / entry for k in range(hold + 1)])


def twin_days(t, hold, n, rng):
    lo = max(FIRST_DAY, t - TWIN_WINDOW)
    hi = min(n - 2 - hold, t + TWIN_WINDOW)
    return rng.integers(lo, hi, N_RANDOM) if hi > lo else np.array([], int)


def _twins(o, h, l, c, days, d, tgt_pct, stop_pct, hold, line_rel, flags, s):
    rets = []
    for t in days:
        e = o[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        res = simulate(o, h, l, c, t, d, e * (1 + d * tgt_pct), None if stop_pct is None else e * (1 - d * stop_pct),
                       hold, line_rel, e, int(s["stop_days"]), flags, _cost(s))
        if res is not None:
            rets.append(res[0])
    return (float(np.mean(rets)), float(np.mean(np.array(rets) > 0))) if rets else (np.nan, np.nan)


def _cost(s):
    return float(s["cost_one_way"]) + float(s["spread_one_way"])


def twin_trades(ticker: str, df: pd.DataFrame, pattern_sigs: list, s: dict, seed: int | None = None) -> pd.DataFrame:
    """One row per pattern trade: its result under each way of managing it, its random twins'
    average on the same days, the edge, and its textbook checks (chk_..., val_...)."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    atr = np.maximum(ind.atr_filled(h, l, c), 1e-9)
    _, _, kr_bull, kr_bear = key_reversal_days(o, h, l, c, v)    # strong key reversal days only (PatternFinder 2.5)
    rng = np.random.default_rng(sum(map(ord, ticker)) if seed is None else seed)   # the same days every run
    sl, k_need, cost = float(s["stop_loss"]), int(s["stop_days"]), _cost(s)
    rows = []
    for sg in sorted(pattern_sigs, key=lambda x: x.idx):
        t, d = int(sg.idx), int(sg.direction)
        if sg.target is None or t + 1 >= n or t < FIRST_DAY:
            continue
        entry = o[t + 1]
        hold = int(max(1, (sg.time_limit if sg.time_limit is not None else t + int(s["default_hold"])) - t))
        tgt_pct = abs(sg.target - entry) / entry
        lr = _line_rel(sg, entry, hold)
        book_stop = entry * (1 - d * sl)
        base = simulate(o, h, l, c, t, d, sg.target, book_stop, hold, lr, entry, k_need, None, cost)
        if base is None:
            continue
        days = twin_days(t, hold, n, rng)
        twin, twin_win = _twins(o, h, l, c, days, d, tgt_pct, sl, hold, lr, None, s)
        row = {"ticker": ticker, "pattern": sg.rule, "bias": "Bullish" if d > 0 else "Bearish",
               "signal_date": f"{df.index[t]:%Y-%m-%d}", "entry_date": f"{df.index[t + 1]:%Y-%m-%d}",
               "exit_date": f"{df.index[base[1]]:%Y-%m-%d}", "exit_reason": base[2], "days_held": int(base[1] - t),
               "entry": round(entry, 4), "target": round(float(sg.target), 4), "stop": round(book_stop, 4),
               "return_pct": 100 * base[0], "r_multiple": base[0] / sl,
               "random_return_pct": 100 * twin, "random_win_rate": twin_win, "edge_pct": 100 * (base[0] - twin),
               "twins": len(days)}
        flags = kr_bear if d > 0 else kr_bull
        kr = simulate(o, h, l, c, t, d, sg.target, book_stop, hold, lr, entry, k_need, flags, cost)
        kr_twin, _ = _twins(o, h, l, c, days, d, tgt_pct, sl, hold, lr, flags, s)
        row.update(kr_exit_return_pct=100 * kr[0], kr_exit_random_return_pct=100 * kr_twin,
                   kr_exit_edge_pct=100 * (kr[0] - kr_twin), kr_exit_reason=kr[2])
        chk = sg.extra.get("checks") or {}
        row.update(gap_stop_return_pct=np.nan, gap_stop_random_return_pct=np.nan, gap_stop_edge_pct=np.nan,
                   gap_stop_reason="")
        if (chk.get("breakaway_gap") or {}).get("passed"):
            g = h[t - 1] - STOP_BUFFER * atr[t] if d > 0 else l[t - 1] + STOP_BUFFER * atr[t]
            tighter = (book_stop < g < entry) if d > 0 else (entry < g < book_stop)
            if tighter:
                gr = simulate(o, h, l, c, t, d, sg.target, g, hold, lr, entry, k_need, None, cost)
                if gr is not None:
                    g_twin, _ = _twins(o, h, l, c, days, d, tgt_pct, abs(g - entry) / entry, hold, lr, None, s)
                    row.update(gap_stop_return_pct=100 * gr[0], gap_stop_random_return_pct=100 * g_twin,
                               gap_stop_edge_pct=100 * (gr[0] - g_twin), gap_stop_reason=gr[2])
        row.update(pattern_stop_price=np.nan, pattern_stop_return_pct=np.nan, pattern_stop_random_return_pct=np.nan,
                   pattern_stop_edge_pct=np.nan, pattern_stop_reason="")
        ps = pattern_stop(sg, h, l, atr)
        if ps is not None:
            pr = simulate(o, h, l, c, t, d, sg.target, ps, hold, None, entry, k_need, None, cost)
            if pr is not None:
                p_twin, _ = _twins(o, h, l, c, days, d, tgt_pct, abs(ps - entry) / entry, hold, None, None, s)
                row.update(pattern_stop_price=round(ps, 4), pattern_stop_return_pct=100 * pr[0],
                           pattern_stop_random_return_pct=100 * p_twin, pattern_stop_edge_pct=100 * (pr[0] - p_twin),
                           pattern_stop_reason=pr[2])
        if chk:
            row.update(flat_columns(chk))
        rows.append(row)
    return pd.DataFrame(rows)
