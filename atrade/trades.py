"""Framework 8 - trade management.

Every signal is entered at the NEXT day's open. Pattern trades then close at
whichever comes first:
  1. Target hit  - close reaches the price target            -> "target"
  2. Time limit  - the time-limit day is reached              -> "time"
  3. Stopped out - close back across the broken line for
                   `stop_days` consecutive days, or a loss of
                   `stop_loss` (4%) from the entry price        -> "stop"
"""
from __future__ import annotations

import math

import numpy as np

from .signals import Signal


def pattern_trade(sig: Signal, open_: np.ndarray, close: np.ndarray, s: dict) -> dict | None:
    n = len(close)
    e = sig.idx + 1
    if e >= n or np.isnan(open_[e]):
        return {"status": "pending", "entry_idx": None}
    d = sig.direction
    entry = float(open_[e])
    tl = sig.time_limit if sig.time_limit is not None else sig.idx + int(s["default_hold"])
    tl = max(tl, e)
    k_need = int(s["stop_days"])
    sl = float(s["stop_loss"])
    back = 0
    for u in range(e, min(n, tl + 1)):
        c = close[u]
        if np.isnan(c):
            continue
        if sig.target is not None and ((d > 0 and c >= sig.target) or (d < 0 and c <= sig.target)):
            return _done(e, entry, u, c, d, "target")
        if (d > 0 and c <= entry * (1 - sl)) or (d < 0 and c >= entry * (1 + sl)):
            return _done(e, entry, u, c, d, "stop")
        lv = sig.line_value(u)
        if lv is not None:
            crossed_back = (d > 0 and c < lv) or (d < 0 and c > lv)
            back = back + 1 if crossed_back else 0
            if back >= k_need:
                return _done(e, entry, u, c, d, "stop")
        if u == tl:
            return _done(e, entry, u, c, d, "time")
    last = n - 1
    return {"status": "open", "entry_idx": e, "entry_price": entry, "exit_idx": None,
            "last_idx": last, "unrealised": d * math.log(close[last] / entry) if close[last] > 0 else np.nan,
            "outcome": "open"}


def _done(e, entry, u, c, d, outcome):
    return {"status": "closed", "entry_idx": e, "entry_price": entry, "exit_idx": u,
            "exit_price": float(c), "outcome": outcome, "ret": d * math.log(c / entry)}


def trade_plan(sig: Signal | None, direction: int, last_close: float, last_idx: int, s: dict) -> dict:
    """The exit plan attached to today's Buy or Sell rating."""
    sl = float(s["stop_loss"])
    plan = {"direction": direction, "entry": "Next trading day's open",
            "stop_loss_pct": sl, "stop_days": int(s["stop_days"])}
    if sig is not None and sig.target is not None:
        plan["source"] = sig.rule
        plan["target"] = sig.target
        plan["time_limit_idx"] = sig.time_limit
        plan["line_today"] = sig.line_value(last_idx)
        plan["kind"] = "pattern"
    else:
        plan["source"] = sig.rule if sig is not None else "Rating engine"
        plan["target"] = None
        plan["time_limit_idx"] = last_idx + int(s["default_hold"])
        plan["line_today"] = sig.line_value(last_idx) if sig is not None else None
        plan["kind"] = "fixed holding period"
    plan["approx_stop_price"] = last_close * (1 - sl) if direction > 0 else last_close * (1 + sl)
    return plan
