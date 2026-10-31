"""The common record that every framework produces."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Signal:
    rule: str                 # rule name, e.g. "MACD 12/26/9"
    group: str                # "trend", "reversal" or "sentiment"
    kind: str                 # "crossover", "breakout", "pattern", "zone", "gauge", "sentiment", "benchmark"
    direction: int            # +1 buy, -1 sell
    idx: int                  # day number on which the signal is known (entry is next day's open)
    weight: float = 1.0       # rating-engine weight inside the group
    live_until: int | None = None   # last day (inclusive) the vote counts in the rating
    vol_adjust: bool = True   # multiply by the Framework 5 volume weight?
    text: str = ""            # plain-language reason
    # Framework 8 exit plan (patterns only)
    target: float | None = None
    time_limit: int | None = None
    line: tuple[float, float] | None = None   # broken line: value on day s = slope*s + intercept
    points: list = field(default_factory=list)  # [(idx, price, label)] for drawing
    extra: dict = field(default_factory=dict)

    def line_value(self, s: int) -> float | None:
        if self.line is None:
            return None
        return self.line[0] * s + self.line[1]


@dataclass
class Setup:
    """A chart pattern whose shape is complete, judged before (or without) a breakout.

    A Setup is a different kind of record from a Signal on purpose: the rating
    engine and the back-test only ever see Signals, so a forming pattern can
    never vote. A pattern that breaks out has both a Setup (its history) and a
    Signal (the breakout itself, which votes and is back-tested).
    """
    rule: str                 # pattern name, the same as its Signal's rule
    direction: int            # +1 expected to break up, -1 down, 0 either way
    known: int                # day the shape is confirmed (its last turning point + w)
    deadline: int             # last day (inclusive) on which a break still counts
    breaks: list              # [(direction, line)] lines whose break would be a signal
    height: float             # measured move added to the broken line for the target
    status: str = ""          # forming.FORMING, BROKE_OUT, FAILED or EXPIRED (as of the last day)
    resolved: int | None = None   # day the status was decided (None while forming)
    broke_dir: int = 0        # direction of the break, when the status is BROKE_OUT
    reason: str = ""          # plain-language reason for a Failed or Expired status
    fixed_target: float | None = None   # used instead of line + height when set (wedges)
    points: list = field(default_factory=list)  # [(idx, price, label)] for drawing
    extra: dict = field(default_factory=dict)

    def target_if(self, d: int, s: int) -> float | None:
        """Price target if the pattern broke in direction d on day s."""
        for dd, ln in self.breaks:
            if dd == d:
                if self.fixed_target is not None:
                    return self.fixed_target
                return ln[0] * s + ln[1] + d * self.height
        return None
