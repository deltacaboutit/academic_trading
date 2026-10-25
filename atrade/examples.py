"""Textbook example shapes for every chart pattern.

These clean, made-up price paths are used in two places:
  * the tests check that the tool's own detectors find each one;
  * the Guide tab draws its pictures from them, using what the detectors found.
So the pictures show exactly the lines and targets the tool uses.
Bearish versions are the bullish shapes turned upside down (price -> 200 - price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def path(knots):
    """Straight lines between (day, price) points."""
    days = [k[0] for k in knots]; px = [k[1] for k in knots]
    return np.interp(np.arange(days[-1] + 1), days, px)


def _flip(c):
    return 200 - np.asarray(c, float)


_HS = [(0, 80), (20, 100), (40, 90), (60, 110), (80, 90), (100, 100), (120, 80)]
_DOUBLE = [(0, 80), (30, 100), (55, 85), (80, 100.5), (100, 90), (115, 80)]
_TRIPLE = [(0, 80), (30, 100), (55, 88), (80, 99.9), (105, 88.5), (130, 100.3), (150, 85)]
_ASC = [(0, 70), (10, 100), (25, 80), (40, 100.5), (55, 85), (70, 99.8), (80, 90), (100, 115)]
_RECT = [(0, 80), (10, 110), (25, 100), (40, 110.5), (55, 100.5), (70, 109.8), (78, 103), (95, 120)]
_SYM = [(0, 70), (10, 130), (25, 80), (40, 123), (55, 87), (70, 116), (78, 100), (95, 125)]
_FLAG = [(0, 50), (40, 100), (43, 96), (46, 98), (49, 94), (60, 110)]
_PENNANT = [(0, 50), (30, 60), (40, 100), (44, 92), (48, 98), (52, 93.5), (54, 97.5), (64, 110)]
_WEDGE = [(0, 60), (15, 64), (20, 70), (25, 100), (35, 90), (45, 94.2), (55, 87.3), (62, 90.5), (80, 100)]


def _cup():
    t = np.arange(0, 81)
    cup = 100 - 25 * np.sin(np.pi * t / 80)
    return np.concatenate([np.linspace(85, 99, 15), [100.5], cup[1:-1], [100],
                           np.linspace(99, 94, 8)[1:], np.linspace(94, 105, 10)[1:], np.linspace(105, 112, 20)[1:]])


def _bowl():
    t = np.arange(0, 61)
    bowl = 100 - 20 * np.sin(np.pi * t / 60)
    p = np.concatenate([np.linspace(90, 100, 15)[:-1] + np.r_[np.zeros(13), 0.5], bowl, np.linspace(100, 110, 20)])
    p[14] = 100.5
    return p


CASES = {
    "Head-and-shoulders top": path(_HS),
    "Inverse head-and-shoulders": _flip(path(_HS)),
    "Double top": path(_DOUBLE),
    "Double bottom": _flip(path(_DOUBLE)),
    "Triple top": path(_TRIPLE),
    "Triple bottom": _flip(path(_TRIPLE)),
    "Bull flag": path(_FLAG),
    "Bear flag": _flip(path(_FLAG)),
    "Bull pennant": path(_PENNANT),
    "Bear pennant": _flip(path(_PENNANT)),
    "Falling wedge": path(_WEDGE),
    "Rising wedge": _flip(path(_WEDGE)),
    "Rounding bottom": _bowl(),
    "Rounding top": _flip(_bowl()),
    "Ascending triangle": path(_ASC),
    "Descending triangle": _flip(path(_ASC)),
    "Symmetrical triangle": path(_SYM),
    "Rectangle": path(_RECT),
    "Cup and handle": _cup(),
}


def island(top=True):
    """Closes, highs and lows for an island: a rise, a gap up, three days stranded, a gap down."""
    c = np.r_[np.linspace(80, 100, 30), [104, 105, 104.5], np.linspace(98, 90, 10)]
    h, lo = c + 0.5, c - 0.5
    return (c, h, lo) if top else (200 - c, 200 - lo, 200 - h)


GAP_CASES = {"Island top": lambda: island(True), "Island bottom": lambda: island(False)}


def example_chart(rule: str) -> tuple[pd.DataFrame, object]:
    """A price table for the rule's example and the Signal the detectors found in it."""
    from .config import DEFAULTS
    from .patterns import all_patterns
    if rule in GAP_CASES:
        c, h, lo = GAP_CASES[rule]()
    else:
        c = CASES[rule]; h = lo = None
    sigs = [x for x in all_patterns(c, DEFAULTS, np.zeros(len(c)), h, lo) if x.rule == rule]
    df = pd.DataFrame({"Close": c, "High": c if h is None else h, "Low": c if lo is None else lo}, index=pd.bdate_range("2024-01-01", periods=len(c)))
    return df, (sigs[0] if sigs else None)
