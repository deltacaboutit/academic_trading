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
