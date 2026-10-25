"""Plain-English text shown in the app: indicator sentences, one pattern's
explanation, and a summary of the stock.

Adapted from PatternFinder 2.5 (pattern_core/report.py and
indicators.describe_indicators). Everything is worked out from the result of
analysis.analyze(), so the text always matches the numbers on the other tabs.
The text is Markdown, which the app shows with st.markdown.
"""
from __future__ import annotations

import numpy as np

from . import indicators as ind
from .forming import TRADE_STATUS, forming_now, signal_status
from .pattern_info import DISCLAIMER, PATTERN_INFO, STATUS_HELP, STOP_RULE
from .signals import Setup, Signal

RECENT_DAYS = 10    # a breakout this many trading days ago or fewer counts as "recent" (PatternFinder 2.5)


def price(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.2f}"


def change(a, b) -> str:
    """Percentage change from a to b, with a sign."""
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) or a == 0:
        return ""
    return f"{(b - a) / a * 100:+.1f}%"


def _date(res, i):
    idx = res["df"].index
    if i is None:
        return "n/a"
    if 0 <= i < len(idx):
        return f"{idx[i]:%d %b %Y}"
    import pandas as pd
    return f"{idx[-1] + pd.tseries.offsets.BDay(i - (len(idx) - 1)):%d %b %Y}"


def _days_ago(n, i):
    k = n - 1 - i
    return "today" if k == 0 else "yesterday" if k == 1 else f"{k} trading days ago"


def _cross(a, b, lookback):
    """How many days ago line a last crossed line b, and which way. None if not within `lookback` days."""
    d = np.sign(np.asarray(a, float) - np.asarray(b, float))
    n = len(d)
    for k in range(min(lookback, n - 1)):
        now, before = d[n - 1 - k], d[n - 2 - k]
        if np.isnan(now) or np.isnan(before):
            return None
        if now != before and now != 0:
            return k, ("up" if now > 0 else "down")
    return None


# ------------------------------------------------------------ indicators ----

def indicator_sentences(res) -> list[str]:
    """The latest indicator readings, one plain-English sentence each."""
    df, s = res["df"], res["settings_used"]
    c = df["Close"].to_numpy(float)
    n = len(c)
    px = c[-1]
    out = []
    s50, s200 = ind.sma(c, 50), ind.sma(c, 200)
    a, b = s50[-1], s200[-1]
    if np.isfinite(a) and np.isfinite(b):
        if px > a > b:
            out.append(f"**Trend: up.** The price ({price(px)}) is above its 50-day average ({price(a)}), which is "
                       f"above its 200-day average ({price(b)}). This is the classic picture of an uptrend.")
        elif px < a < b:
            out.append(f"**Trend: down.** The price ({price(px)}) is below its 50-day average ({price(a)}), which is "
                       f"below its 200-day average ({price(b)}). This is the classic picture of a downtrend.")
        else:
            out.append(f"**Trend: mixed.** The price ({price(px)}), the 50-day average ({price(a)}) and the 200-day "
                       f"average ({price(b)}) are not lined up in one direction.")
        x = _cross(s50, s200, 20)
        if x:
            k, way = x
            when = _days_ago(n, n - 1 - k)
            out.append(f"A 'golden cross' happened {when}: the 50-day average moved above the 200-day average. "
                       "Many traders see this as a positive sign." if way == "up" else
                       f"A 'death cross' happened {when}: the 50-day average moved below the 200-day average. "
                       "Many traders see this as a negative sign.")
    st = int(res["state"][-1])
    out.append("The tool's own trend state (Framework 3) is " +
               {1: "**uptrend**", -1: "**downtrend**", 0: "**sideways**"}[st] +
               f": it compares the price with its {s['trend_state_ma']}-day average over the last "
               f"{s['trend_state_days']} days.")
    r = res["gauges"]["rsi"][-1]
    lo, hi = float(s["rsi_low"]), float(s["rsi_high"])
    if np.isfinite(r):
        if r >= hi:
            out.append(f"RSI is {r:.0f} (above {hi:.0f}): 'overbought'. The price has risen quickly and may pause, "
                       "although strong trends can stay overbought for a while.")
        elif r <= lo:
            out.append(f"RSI is {r:.0f} (below {lo:.0f}): 'oversold'. The price has fallen quickly and may bounce, "
                       "although weak stocks can stay oversold for a while.")
        else:
            mood = "slightly positive" if r > 55 else "slightly negative" if r < 45 else "neutral"
            out.append(f"RSI is {r:.0f}, in the normal range ({lo:.0f} to {hi:.0f}). Momentum looks {mood}.")
    m, sg = ind.macd(c, int(s["macd_fast"]), int(s["macd_slow"]), int(s["macd_signal"]))
    if np.isfinite(m[-1]) and np.isfinite(sg[-1]):
        x = _cross(m, sg, 5)
        if x:
            k, way = x
            out.append(f"MACD crossed {'above' if way == 'up' else 'below'} its signal line {_days_ago(n, n - 1 - k)}. "
                       f"Short-term momentum is turning {'up' if way == 'up' else 'down'}.")
        else:
            out.append(f"MACD is {'above' if m[-1] > sg[-1] else 'below'} its signal line, so short-term momentum is "
                       f"pointing {'up' if m[-1] > sg[-1] else 'down'}.")
    bb = res["gauges"]["bb"]
    if np.isfinite(bb["up"][-1]):
        if px > bb["up"][-1]:
            out.append(f"The close is above the upper Bollinger Band ({price(bb['up'][-1])}), so the price is "
                       "unusually stretched upward.")
        elif px < bb["low"][-1]:
            out.append(f"The close is below the lower Bollinger Band ({price(bb['low'][-1])}), so the price is "
                       "unusually stretched downward.")
    rv = res["rv"][-1]
    if np.isfinite(rv):
        if rv >= float(s["rv_full"]):
            out.append(f"Today's volume was {rv:.1f} times its {s['rv_days']}-day average. Heavy volume means many "
                       "traders were active, so today's signals count at full weight.")
        elif rv < float(s["rv_half"]):
            out.append(f"Today's volume was light ({rv:.1f} times its {s['rv_days']}-day average), so today's "
                       "signals are not confirmed by volume.")
    return out


# --------------------------------------------------------------- patterns ----

def _info_lines(rule):
    info = PATTERN_INFO.get(rule, {})
    if not info:
        return []
    return ["", "**What it looks like.** " + info["looks"], "", "**What it may mean.** " + info["means"],
            "", "**The tool's rule.** " + info["rule"]]


def describe_signal(sg: Signal, res) -> str:
    """The explanation of a pattern that broke out."""
    df, s = res["df"], res["settings_used"]
    c = df["Close"].to_numpy(float)
    n = len(c)
    info = PATTERN_INFO.get(sg.rule, {})
    status = signal_status(sg)
    tr = sg.extra.get("trade") or {}
    way = "above" if sg.direction > 0 else "below"
    L = [f"### {sg.rule}", f"{info.get('category', '')} pattern · usual direction: {info.get('bias', '')}", "",
         f"**Status: {status}.** {STATUS_HELP.get(status, '')}", "",
         f"The close went {way} the pattern's line on **{_date(res, sg.idx)}** ({price(c[sg.idx])}), which made a "
         f"{'buy' if sg.direction > 0 else 'sell'} signal. A trade would start at the next day's open."]
    rv = res["rv"][sg.idx]
    if np.isfinite(rv):
        if rv >= float(s["rv_full"]):
            L.append(f"Volume that day was {rv:.1f} times its {s['rv_days']}-day average, so the signal counts at "
                     "full weight. Classic chart books see heavy volume on a breakout as a sign it is more likely "
                     "to be genuine.")
        elif rv >= float(s["rv_half"]):
            L.append(f"Volume that day was {rv:.1f} times its {s['rv_days']}-day average (about normal), so the "
                     "signal counts at half weight.")
        else:
            L.append(f"Volume that day was below average ({rv:.1f} times), so the signal is not confirmed by volume "
                     "and counts for nothing in the rating.")
    if sg.extra.get("trend_before"):
        L.append(f"The trend before the pattern was: {sg.extra['trend_before']}.")
    L += _info_lines(sg.rule)
    L += ["", "**The numbers**"]
    entry = tr.get("entry_price")
    if entry is not None:
        L.append(f"- Entry (next day's open): {price(entry)} on {_date(res, tr['entry_idx'])}")
    if sg.target is not None:
        L.append(f"- Target: {price(sg.target)}" + (f" ({change(entry, sg.target)} from the entry)" if entry else "")
                 + (f". {info['target']}" if info else ""))
    if sg.time_limit is not None:
        L.append(f"- Time limit: {_date(res, sg.time_limit)}")
    L.append(f"- Stop: {STOP_RULE}")
    if entry is not None:
        sl = float(s["stop_loss"])
        stop_px = entry * (1 - sl) if sg.direction > 0 else entry * (1 + sl)
        L.append(f"  The 4% loss level is {price(stop_px)}.")
    if tr.get("status") == "closed":
        L.append(f"- The trade ended on {_date(res, tr['exit_idx'])} at {price(tr['exit_price'])}: "
                 f"{TRADE_STATUS[tr['outcome']].lower()}, a return of {tr['ret'] * 100:+.1f}% before costs.")
    elif tr.get("status") == "open":
        lv = sg.line_value(n - 1)
        L.append(f"- The trade is still open. Latest close {price(c[-1])}: "
                 + (f"{change(c[-1], sg.target)} to the target" if sg.target is not None else "")
                 + (f", and the broken line is at {price(lv)} today." if lv is not None else "."))
    L += ["", DISCLAIMER]
    return "\n".join(L)


def describe_setup(u: Setup, res) -> str:
    """The explanation of a pattern that has not broken out (forming, failed or expired)."""
    n = len(res["df"])
    c = res["df"]["Close"].to_numpy(float)
    info = PATTERN_INFO.get(u.rule, {})
    t = n - 1 if u.status == "Forming" else min(u.resolved if u.resolved is not None else n - 1, n - 1)
    L = [f"### {u.rule}", f"{info.get('category', '')} pattern · usual direction: {info.get('bias', '')}", "",
         f"**Status: {u.status}.** {STATUS_HELP.get(u.status, '')}", "",
         f"The shape was complete and confirmed on **{_date(res, u.known)}** (its last turning point is only known "
         f"once the next few days have passed)."]
    if u.status in ("Failed", "Expired") and u.resolved is not None:
        L.append(f"On {_date(res, u.resolved)} it {u.reason.replace('closed', 'had closed') if u.status == 'Failed' else 'expired: ' + u.reason}.")
    L += _info_lines(u.rule)
    L += ["", "**The numbers**" + (" (for a break today)" if u.status == "Forming" else
                                   f" (where the lines were on {_date(res, t)})")]
    for d, ln in u.breaks:
        lv = ln[0] * t + ln[1]
        tg = u.target_if(d, t)
        L.append(f"- A close {'above' if d > 0 else 'below'} **{price(lv)}** would be a breakout "
                 f"{'up' if d > 0 else 'down'}, with a target of {price(tg)} ({change(lv, tg)}).")
    if info:
        L.append(f"- How the target is worked out: {info['target']}")
    L.append(f"- The breakout must come by **{_date(res, u.deadline)}**.")
    if u.status == "Forming":
        L.append(f"- Latest close: {price(c[-1])}. A forming pattern has no vote in the rating; it only becomes a "
                 "signal if it breaks out in time.")
    L += ["", DISCLAIMER]
    return "\n".join(L)


def describe(item, res) -> str:
    return describe_signal(item, res) if isinstance(item, Signal) else describe_setup(item, res)


# ---------------------------------------------------------------- summary ----

def summary_text(res) -> str:
    """A short plain-English summary of the stock: rating, indicators, patterns and their track record."""
    df = res["df"]
    n = len(df)
    c = df["Close"].to_numpy(float)
    L = [f"**{res['ticker']}**: {n:,} trading days from {_date(res, 0)} to {_date(res, n - 1)}. "
         f"Latest close {price(c[-1])}.", "",
         f"**Today's rating: {res['rating_now']}.** The rating combines trend, reversal and sentiment signals "
         "(Framework 7). The Rating tab shows every signal that counts today.", "", "**What the indicators say**"]
    L += [f"- {x}" for x in indicator_sentences(res)]
    L += ["", "**Chart patterns**"]
    fm = res.get("forming") or forming_now(res.get("setups", []))
    recent = [sg for sg in res["patterns"] if sg.idx >= n - RECENT_DAYS]
    if fm:
        L.append(f"{len(fm)} pattern(s) are forming now (complete shape, no breakout yet; they do not vote):")
        for u in fm[:6]:
            parts = [f"{'above' if d > 0 else 'below'} {price(ln[0] * (n - 1) + ln[1])}" for d, ln in u.breaks]
            L.append(f"- {u.rule}: would break out on a close {' or '.join(parts)}, by {_date(res, u.deadline)}.")
    if recent:
        L.append(f"{len(recent)} pattern(s) broke out in the last {RECENT_DAYS} trading days:")
        for sg in sorted(recent, key=lambda x: -x.idx):
            L.append(f"- {sg.rule} ({'buy' if sg.direction > 0 else 'sell'}) on {_date(res, sg.idx)}: "
                     f"{signal_status(sg).lower()}, target {price(sg.target)}.")
    if not fm and not recent:
        L.append("No pattern is forming or has broken out in the last "
                 f"{RECENT_DAYS} trading days.")
    done = [(sg.extra.get("trade") or {}).get("outcome") for sg in res["patterns"]]
    done = [o for o in done if o in ("target", "stop", "time")]
    if done:
        L += ["", "**Track record on this stock**",
              f"Of the {len(done)} earlier pattern trades that have ended, {done.count('target')} reached the target, "
              f"{done.count('stop')} were stopped out and {done.count('time')} reached the time limit. The Back-test "
              "tab shows whether this beat simply owning the stock; a small sample like this is only a rough guide."]
    L += ["", DISCLAIMER]
    return "\n".join(L)
