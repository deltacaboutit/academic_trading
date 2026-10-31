"""Pattern statuses: patterns that are still forming, and what became of them.

A chart pattern goes through these stages (the names follow PatternFinder 2.5):
  Forming          the shape is complete and confirmed, but price has not yet
                   closed beyond its line, and the deadline has not passed
  Failed           before any breakout, price closed across the pattern's other
                   line or beyond its extreme (for example above the head of a
                   head-and-shoulders top), or broke out too early to count
  Expired          the deadline passed without a breakout
  Confirmed        price broke out (this is the pattern's Signal) and the trade is still open
  Target reached / Stopped out / Time limit reached
                   how the trade ended, using the trade rules in trades.py

Forming, Failed and Expired come from Setup records (see patterns._track).
Setups never vote in the rating and are never back-tested: only the Signal
on the breakout day does that. Every status is judged with prices up to the
day it is decided, so there is no look-ahead.

Tool choice: a "Failed" status for closing beyond the pattern's extreme only
labels the setup. It does not change the book's signals, so a later break of
the line still counts as a signal under the book's rules.
"""
from __future__ import annotations

import pandas as pd

from .signals import Setup, Signal

FORMING, BROKE_OUT, FAILED, EXPIRED = "Forming", "Broke out", "Failed", "Expired"
CONFIRMED, TARGET, STOPPED, TIME = "Confirmed", "Target reached", "Stopped out", "Time limit reached"
TRADE_STATUS = {"target": TARGET, "stop": STOPPED, "time": TIME, "open": CONFIRMED, "pending": CONFIRMED}


def unique(setups: list[Setup]) -> list[Setup]:
    """One Setup per shape. Overlapping turning-point windows (and cups with
    different left rims) can find the same shape more than once; keep the best fit."""
    best: dict = {}
    for u in setups:
        key = (u.rule, u.known, u.status, u.resolved, u.broke_dir)
        old = best.get(key)
        if old is None or u.extra.get("fit", 0) > old.extra.get("fit", 0):
            best[key] = u
    return sorted(best.values(), key=lambda u: (u.known, u.rule))


def forming_now(setups: list[Setup]) -> list[Setup]:
    """Patterns still forming on the last day, newest first. Where several
    forming shapes of the same pattern overlap, keep the best fit (cups), or
    else the newest shape (tool choice)."""
    cand = [u for u in unique(setups) if u.status == FORMING]
    cand.sort(key=lambda u: (u.extra.get("fit", 0), u.known, u.height), reverse=True)
    keep: list[Setup] = []
    for u in cand:
        a, b = u.points[0][0], u.points[-1][0]
        if any(k.rule == u.rule and a <= k.points[-1][0] and k.points[0][0] <= b for k in keep):
            continue
        keep.append(u)
    return sorted(keep, key=lambda u: u.known, reverse=True)


def signal_status(sg: Signal) -> str:
    """Status of a pattern that broke out, from its trade (Framework 8)."""
    tr = sg.extra.get("trade") or {}
    return TRADE_STATUS.get(tr.get("outcome") or tr.get("status"), CONFIRMED)


def status_counts(setups: list[Setup], pattern_sigs: list[Signal]) -> pd.DataFrame:
    """For each pattern: how many confirmed shapes broke out, failed, expired or
    are still forming, and how the breakouts' trades ended."""
    rows = {}
    for u in unique(setups):
        r = rows.setdefault(u.rule, {"rule": u.rule, "shapes": 0, FORMING: 0, BROKE_OUT: 0, FAILED: 0, EXPIRED: 0})
        r["shapes"] += 1
        r[u.status] += 1
    for sg in pattern_sigs:               # rounding patterns and islands have signals but no setups
        rows.setdefault(sg.rule, {"rule": sg.rule, "shapes": 0, FORMING: 0, BROKE_OUT: 0, FAILED: 0, EXPIRED: 0})
    out = pd.DataFrame(list(rows.values()), columns=["rule", "shapes", BROKE_OUT, FAILED, EXPIRED, FORMING])
    if out.empty:
        return out
    trades = pd.DataFrame([{"rule": sg.rule, "status": signal_status(sg)} for sg in pattern_sigs],
                          columns=["rule", "status"])
    for name in (TARGET, STOPPED, TIME, CONFIRMED):
        k = trades[trades["status"] == name].groupby("rule").size()
        out[name] = out["rule"].map(k).fillna(0).astype(int)
    return out.sort_values("shapes", ascending=False).reset_index(drop=True)
