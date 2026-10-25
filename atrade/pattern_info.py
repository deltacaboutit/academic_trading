"""Plain-English descriptions of every chart pattern, the statuses and the words used.

Adapted from PatternFinder 2.5 (pattern_core/pattern_info.py) and rewritten to
match this tool's own rules: its turning points, deadlines, targets and the
single stop rule from Framework 8.

Each pattern has:
  category      Reversal (the trend may turn), Continuation (the trend may carry
                on) or Either way (it can break in both directions)
  bias          Bullish (expects a rise), Bearish (expects a fall) or Neutral
  looks         what the shape looks like on a chart
  means         what traders usually take it to mean
  rule          the tool's exact rule, in words
  target        how the price target is worked out
  pattern_stop  where PatternFinder 2.5 and the chart books put a stop-loss for
                this pattern. The tool itself uses one stop rule for every
                pattern (STOP_RULE below); the pattern's own stop is tested as
                a trade-management variant in the research method.
"""
from __future__ import annotations

STOP_RULE = ("Close the trade if the price closes back across the broken line for 2 days in a row, or if the "
             "trade loses 4% from the entry price (Framework 8). The same rule is used for every pattern.")

PATTERN_INFO = {
    # ------------------------------------------------------------ reversals --
    "Head-and-shoulders top": dict(
        category="Reversal", bias="Bearish",
        looks="Three peaks in a row. The middle peak (the 'head') is the highest, and the two outer peaks "
              "(the 'shoulders') are lower and roughly equal. A line joining the two dips between the peaks "
              "is called the 'neckline'.",
        means="Buyers pushed the price to a new high (the head) but could not do it again (the right shoulder). "
              "A close below the neckline suggests the uptrend is turning into a downtrend.",
        rule="Five turning points (peak, dip, peak, dip, peak) found with a 7-day window. The head is the highest; "
             "each shoulder is above the middle of the other shoulder and its dip; the two halves are not more "
             "than 2.5 times as long as each other (book, Table 5.2). The close must break the neckline within "
             "one pattern width after the right shoulder.",
        target="The head's height above the neckline, subtracted from the neckline on the breakout day.",
        pattern_stop="Just above the right shoulder.",
    ),
    "Inverse head-and-shoulders": dict(
        category="Reversal", bias="Bullish",
        looks="Three troughs in a row. The middle one (the 'head') is the lowest, and the two outer troughs "
              "(the 'shoulders') are higher and roughly equal. The line joining the two bounces between them "
              "is the 'neckline'.",
        means="Sellers pushed the price to a new low but could not do it again. A close above the neckline "
              "suggests a downtrend is turning into an uptrend.",
        rule="The head-and-shoulders top turned upside down.",
        target="The head's depth below the neckline, added to the neckline on the breakout day.",
        pattern_stop="Just below the right shoulder.",
    ),
    "Double top": dict(
        category="Reversal", bias="Bearish",
        looks="Two peaks at about the same price with a dip between them, shaped like the letter 'M'.",
        means="The price hit the same ceiling twice and was pushed back both times. A close below the dip "
              "between the peaks suggests the uptrend is over.",
        rule="Peak, dip, peak found with a 15-day window. The peaks are within 4% of each other and the dip is at "
             "least 10% below the first peak (book). The close must break below the dip within one pattern width.",
        target="The higher peak's height above the dip, subtracted from the dip.",
        pattern_stop="Just above the higher of the two peaks.",
    ),
    "Double bottom": dict(
        category="Reversal", bias="Bullish",
        looks="Two troughs at about the same price with a bounce between them, shaped like the letter 'W'.",
        means="The price found the same floor twice. A close above the bounce between the troughs suggests "
              "the downtrend is over.",
        rule="The double top turned upside down: troughs within 4% of each other, the bounce at least 10% above the first trough.",
        target="The bounce's height above the lower trough, added to the bounce.",
        pattern_stop="Just below the lower of the two troughs.",
    ),
    "Triple top": dict(
        category="Reversal", bias="Bearish",
        looks="Three peaks at about the same price, with two dips between them.",
        means="The price failed three times at the same ceiling. A close below the first dip suggests the "
              "uptrend is over.",
        rule="Five turning points found with a 15-day window. The three peaks are within 4% of each other, the "
             "middle peak is not the highest, and the second dip is within 4% above the first (book). The close "
             "must break below the first dip within one pattern width.",
        target="The higher outer peak's height above the first dip, subtracted from that dip.",
        pattern_stop="Just above the highest of the three peaks.",
    ),
    "Triple bottom": dict(
        category="Reversal", bias="Bullish",
        looks="Three troughs at about the same price, with two bounces between them.",
        means="The price held the same floor three times. A close above the first bounce suggests the "
              "downtrend is over.",
        rule="The triple top turned upside down.",
        target="The first bounce's height above the lower outer trough, added to that bounce.",
        pattern_stop="Just below the lowest of the three troughs.",
    ),
    "Rounding bottom": dict(
        category="Reversal", bias="Bullish",
        looks="A long, smooth, bowl-shaped curve: the price drifts down, flattens out, then drifts back up "
              "to where it started (the 'rim').",
        means="Selling slowly fades and buying slowly returns. A close back at the rim suggests a longer-term "
              "uptrend.",
        rule="From a peak (10-day window) to the first close back at its level, at least 15 days. At least 90% of "
             "the closes must lie inside a band around a circle through the rim and the low (book's circle method).",
        target="The bowl's depth, added to the rim.",
        pattern_stop="Below the lowest price of the last part of the bowl.",
    ),
    "Rounding top": dict(
        category="Reversal", bias="Bearish",
        looks="A long, smooth, upside-down bowl (a dome): the price drifts up, flattens out, then drifts back down.",
        means="Buying slowly fades and selling slowly takes over. A close back at the starting level suggests a "
              "longer-term downtrend.",
        rule="The rounding bottom turned upside down.",
        target="The dome's height, subtracted from the starting level.",
        pattern_stop="Above the highest price of the last part of the dome.",
    ),
    "Cup and handle": dict(
        category="Continuation", bias="Bullish",
        looks="A rounded, U-shaped dip (the 'cup') that climbs back to its starting price, followed by a small "
              "pullback (the 'handle') that stays in the upper half of the cup.",
        means="A slow, orderly recovery followed by a last brief shake-out. A close above the rim suggests a new "
              "move up.",
        rule="Two rims within 4% of each other, 7 to 65 weeks apart, with a cup 12% to 50% deep (O'Neil). The cup "
             "must be round: it passes the circle test, and at least half of its closes are in its lowest 40% "
             "(tool choice). The handle stays in the cup's upper half and lasts at most half the cup's width "
             "and 40 days.",
        target="The cup's depth, added to the rim (Bulkowski).",
        pattern_stop="Just below the lowest point of the handle.",
    ),
    "Island top": dict(
        category="Reversal", bias="Bearish",
        looks="After a rise, the price gaps up (a jump with no trading in between), trades for a few days on its "
              "own 'island', then gaps down, leaving those days stranded above two gaps.",
        means="The first gap is a last burst of buying (an 'exhaustion gap'). The gap down soon after shows the "
              "buyers have given up. Murphy calls this an important turn.",
        rule="A gap up (today's low above yesterday's high) to a new 20-day closing high, then within 10 days a "
             "gap down below every low of the island (Murphy). Needs daily highs and lows.",
        target="The island's own height, subtracted from its lower edge (tool choice; Murphy gives no target).",
        pattern_stop="Just above the island's highest price.",
    ),
    "Island bottom": dict(
        category="Reversal", bias="Bullish",
        looks="After a fall, the price gaps down, trades for a few days on its own 'island', then gaps up, "
              "leaving those days stranded below two gaps.",
        means="The first gap is a last burst of selling. The gap up soon after shows the sellers have given up.",
        rule="The island top turned upside down.",
        target="The island's own depth, added to its upper edge (tool choice).",
        pattern_stop="Just below the island's lowest price.",
    ),
    # --------------------------------------------------------- continuation --
    "Bull flag": dict(
        category="Continuation", bias="Bullish",
        looks="A sharp rise (the 'flagpole') followed by a short, gentle drift down inside two roughly parallel "
              "lines (the 'flag').",
        means="Traders paused to take some profit after a fast rise. A close above the top of the flag suggests "
              "the rise will continue.",
        rule="Peak, dip, lower peak, lower dip, found with a 3-day window, after a rise that started at the lowest "
             "close of the 60 days before. The close must break above the flag's upper line within 20 days of "
             "the flag's start, and the flag must be less than half as long as the pole (book).",
        target="The flagpole's height, added to the upper line on the breakout day.",
        pattern_stop="Just below the lowest point of the flag.",
    ),
    "Bear flag": dict(
        category="Continuation", bias="Bearish",
        looks="A sharp fall (the 'flagpole') followed by a short, gentle drift up inside two roughly parallel lines.",
        means="A brief bounce after a fast fall. A close below the bottom of the flag suggests the fall will continue.",
        rule="The bull flag turned upside down.",
        target="The flagpole's height, subtracted from the lower line on the breakout day.",
        pattern_stop="Just above the highest point of the flag.",
    ),
    "Bull pennant": dict(
        category="Continuation", bias="Bullish",
        looks="A sharp rise followed by a small triangle in which the swings get smaller and smaller.",
        means="A short pause while buyers and sellers balance out. A close above the top line suggests the rise "
              "will continue.",
        rule="Peak, dip, lower peak, higher dip, found with a 2-day window, after a sharp rise. The close must "
             "break above the upper line between half and three-quarters of the way to the point where the two "
             "lines meet, and within 20 days (book).",
        target="The flagpole's height, added to the upper line on the breakout day.",
        pattern_stop="Just below the lowest point of the pennant.",
    ),
    "Bear pennant": dict(
        category="Continuation", bias="Bearish",
        looks="A sharp fall followed by a small triangle in which the swings get smaller and smaller.",
        means="A short pause after a fast fall. A close below the bottom line suggests the fall will continue.",
        rule="The bull pennant turned upside down.",
        target="The flagpole's height, subtracted from the lower line on the breakout day.",
        pattern_stop="Just above the highest point of the pennant.",
    ),
    "Falling wedge": dict(
        category="Continuation", bias="Bullish",
        looks="Both lines slope downward, but the upper line falls faster, so they squeeze together.",
        means="The price is still falling, but each push down is weaker. It usually breaks upward.",
        rule="Peak, dip, peak, dip with both lines falling and the upper one steeper, found with a 7-day window, "
             "after a rise. The close must break above the upper line 20 to 80 days after the wedge starts, "
             "between half and 80% of the way to the point where the lines meet (book).",
        target="Back to the wedge's first peak.",
        pattern_stop="Just below the most recent dip inside the wedge.",
    ),
    "Rising wedge": dict(
        category="Continuation", bias="Bearish",
        looks="Both lines slope upward, but the lower line rises faster, so they squeeze together.",
        means="The price is still rising, but each push up is weaker. It usually breaks downward.",
        rule="The falling wedge turned upside down.",
        target="Back to the wedge's first dip.",
        pattern_stop="Just above the most recent peak inside the wedge.",
    ),
    "Ascending triangle": dict(
        category="Continuation", bias="Bullish",
        looks="A flat ceiling on top and a rising floor underneath (each dip is higher than the last).",
        means="Sellers keep selling at the same price, but buyers step in earlier and earlier. A close above the "
              "flat ceiling usually leads to a move up.",
        rule="Five turning points (7-day window): tops within 4% of each other and rising dips (Lo, Mamaysky & "
             "Wang 2000; Edwards & Magee). The close must break above the ceiling between half and "
             "three-quarters of the way to the point where the lines meet.",
        target="The triangle's height at its widest part, added to the ceiling.",
        pattern_stop="Just below the most recent dip inside the triangle.",
    ),
    "Descending triangle": dict(
        category="Continuation", bias="Bearish",
        looks="A flat floor underneath and a falling ceiling on top (each peak is lower than the last).",
        means="Buyers keep buying at the same price, but sellers get more and more eager. A close below the flat "
              "floor usually leads to a move down.",
        rule="The ascending triangle turned upside down.",
        target="The triangle's height at its widest part, subtracted from the floor.",
        pattern_stop="Just above the most recent peak inside the triangle.",
    ),
    "Rectangle": dict(
        category="Either way", bias="Neutral",
        looks="The price bounces between a flat ceiling and a flat floor, like a box.",
        means="A pause in which buyers and sellers are evenly matched. It can break out either way.",
        rule="Five turning points (7-day window): tops within 4% of each other, dips within 4% of each other, and "
             "the box taller than 4% (tool choice). The close must break out within one pattern width.",
        target="The box's height, added to the ceiling (break up) or subtracted from the floor (break down).",
        pattern_stop="The middle of the box.",
    ),
    "Symmetrical triangle": dict(
        category="Either way", bias="Neutral",
        looks="Peaks keep getting lower and dips keep getting higher, so the two lines squeeze together.",
        means="Neither side is in control and the price is coiling up. The first close outside the lines is "
              "what counts.",
        rule="Five turning points (7-day window): falling peaks and rising dips. The close must break out between "
             "half and three-quarters of the way to the point where the lines meet (Edwards & Magee).",
        target="The triangle's height at its widest part, measured from the broken line in the breakout's direction.",
        pattern_stop="Just beyond the most recent swing on the other side of the triangle.",
    ),
}

STATUS_HELP = {
    "Forming": "The shape is complete, but the price has not yet closed beyond its line. Nothing is confirmed, "
               "and many patterns never break out. A forming pattern has no vote in the rating.",
    "Failed": "Before any breakout, the price went the wrong way: it closed across the pattern's other line or "
              "beyond its extreme, or it broke out too early to count.",
    "Expired": "The price did not break out before the pattern's deadline, so the pattern no longer counts.",
    "Confirmed": "The price closed beyond the line, which made a signal. The trade is still open.",
    "Target reached": "After the breakout, the price reached the target.",
    "Stopped out": "After the breakout, the trade was closed by the stop rule.",
    "Time limit reached": "After the breakout, the trade reached its time limit before the target or the stop.",
}

GLOSSARY = [
    ("Ticker", "The short code for a stock or fund, for example AAPL for Apple or SPY for an S&P 500 fund."),
    ("Candlestick", "Each bar on the chart shows one day. The thin line goes from the lowest to the highest price; "
                    "the thick part goes from the opening to the closing price."),
    ("Volume", "How many shares were traded. The tool compares each day's volume with its 50-day average."),
    ("Moving average", "The average closing price over the last N days. It smooths out the noise. The 50- and "
                       "200-day averages are the most watched."),
    ("RSI", "Relative Strength Index, from 0 to 100. It compares recent gains with recent losses. Above 70 is "
            "called overbought; below 30 is called oversold."),
    ("MACD", "Moving Average Convergence Divergence: the gap between a fast (12-day) and a slow (26-day) "
             "exponential average. When it crosses above its own 9-day average (the signal line), momentum is "
             "turning up."),
    ("Bollinger Bands", "Lines two standard deviations above and below the 20-day average. A close outside them "
                        "means the price is unusually stretched."),
    ("Turning point", "A peak or a trough. The tool marks a day as a peak if its close is higher than every close "
                      "in the w days before and the w days after, so a turning point is only known w days later."),
    ("Breakout", "A close beyond a pattern's key line. This is the day the pattern becomes a signal."),
    ("Forming pattern", "A complete shape that has not broken out yet. It is shown as a heads-up only and has no "
                        "vote in the rating."),
    ("Target", "Where the 'measured move' rule says the price may go after the breakout. It is an estimate, not "
               "a promise."),
    ("Time limit", "The day by which a pattern trade is closed if neither the target nor the stop has been reached."),
    ("Stop-loss", "A rule for getting out when the pattern looks wrong, to limit the loss."),
    ("Gap", "A jump in price with no trading in between: today's lowest price is above yesterday's highest "
            "(or the reverse)."),
    ("Signal", "A rule saying 'buy' or 'sell' on a particular day. Trades start at the next day's open."),
    ("Back-test", "Trying a rule on past prices, in time order, to see how it would have done."),
]

DISCLAIMER = ("This tool is for learning and research, not financial advice. Chart patterns are rules of thumb. "
              "They often fail, and they also appear by chance in completely random prices. The books found that "
              "most technical rules did not reliably beat buy-and-hold after costs.")
