"""Runs Frameworks 1-9 for one stock and returns everything the app displays."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import data as D
from .cleaning import clean
from .gauges import gauge_signals, gauge_values
from .patterns import all_patterns
from .rating import RATING_NAME, live_votes, rating_signals, run_rating
from .sentiment import extreme_info, sentiment_signals
from .trades import pattern_trade, trade_plan
from .trend import alexander_signals, trend_signals, trend_state
from .validation import evaluate_rules, pass_table, pattern_outcomes
from .volume import relative_volume
from .zones import zone_analysis


def _dedupe(sigs):
    seen = {}
    for sg in sigs:
        key = (sg.rule, sg.idx, sg.direction)
        best = seen.get(key)
        if best is None or sg.extra.get("fit", 0) > best.extra.get("fit", 0):
            seen[key] = sg
    return sorted(seen.values(), key=lambda x: x.idx)


def analyze(ticker: str, s: dict, refresh: bool = False, progress=None, prices=None) -> dict:
    def step(msg):
        if progress:
            progress(msg)

    step("Loading prices")
    if prices is None:
        raw, source = D.load_prices(ticker, int(s["history_years"]), refresh)
    else:
        raw, source = prices, "Supplied data"
    step("Framework 1: cleaning the data")
    df, clean_log, quality = clean(raw, s)
    if len(df) < 300:
        raise ValueError(f"Only {len(df)} trading days available; at least 300 are needed.")
    o = df["Open"].to_numpy(float); c = df["Close"].to_numpy(float)
    n = len(c)

    step("Framework 3: trend")
    state = trend_state(c, s)
    sig_trend = trend_signals(df, s)
    sig_alex = alexander_signals(c, float(s["alexander_x"]))

    step("Framework 4: reversals (zones, patterns, gauges)")
    gv = gauge_values(c, s)
    sig_gauge = gauge_signals(c, s, gv)
    sig_zone, zone_events, zone_hist = zone_analysis(c, s)
    sig_pat = _dedupe(all_patterns(c, s, state))
    for sg in sig_pat:
        tr = pattern_trade(sg, o, c, s)
        sg.extra["trade"] = tr
        if tr and tr.get("status") == "closed":
            sg.live_until = tr["exit_idx"]
        else:
            sg.live_until = max(sg.time_limit or sg.idx, sg.idx)

    step("Framework 5: volume")
    rv = relative_volume(df["Volume"].to_numpy(float), int(s["rv_days"]))

    step("Framework 6: options sentiment")
    try:
        series = D.load_sentiment(ticker, df.index, int(s["history_years"]))
    except Exception:
        series = {}
    sig_sent, sent_status = sentiment_signals(series, df, s)
    extremes = {k: extreme_info(v) for k, v in series.items()}

    all_sigs = sig_trend + sig_gauge + sig_zone + sig_pat + sig_sent

    step("Framework 9: back-testing every rule")
    by_rule = defaultdict(list)
    for sg in all_sigs + sig_alex:
        by_rule[sg.rule].append(sg)
    res = evaluate_rules(by_rule, df, s)
    passes = pass_table(res, s)
    allowed = None
    if s.get("only_passed_rules"):
        allowed = set(passes.loc[passes["verdict"].str.startswith("Passed"), "rule"])

    step("Framework 7: rating engine")
    rating_df, index = run_rating(n, all_sigs, state, rv, s, allowed)
    rating_df.index = df.index
    rsig = rating_signals(rating_df["rating"].to_numpy())
    res_rating = evaluate_rules({"Official rating (variant A)": rsig}, df, s)
    res = pd.concat([res, res_rating], ignore_index=True)
    passes = pass_table(res, s)

    t = n - 1
    votes = live_votes(t, index, int(state[t]), rv, s)
    cur = int(rating_df["rating"].iloc[-1])

    step("Framework 8: trade plan")
    plan = None
    if cur != 0:
        same = [sg for (sg, cval, _) in votes if sg.direction == cur and cval != 0]
        pats = sorted([x for x in same if x.kind == "pattern"], key=lambda x: x.idx)
        lead = pats[-1] if pats else (sorted(same, key=lambda x: (x.weight, x.idx))[-1] if same else None)
        plan = trade_plan(lead, cur, float(c[-1]), t, s)
        if plan.get("time_limit_idx") is not None:
            k = plan["time_limit_idx"]
            plan["time_limit_date"] = (df.index[k].date() if k < n else
                                       (df.index[-1] + pd.tseries.offsets.BDay(k - (n - 1))).date())
        # when did the current rating start?
        r = rating_df["rating"].to_numpy()
        j = t
        while j > 0 and r[j - 1] == cur:
            j -= 1
        plan["since"] = df.index[j].date()

    return {
        "ticker": ticker.upper(), "source": source, "df": df, "clean_log": clean_log, "quality": quality,
        "state": state, "rv": rv, "gauges": gv,
        "signals": all_sigs, "alexander": sig_alex, "patterns": sig_pat,
        "zone_events": zone_events, "zones_now": zone_hist.get(n - 1, []),
        "sentiment_series": series, "sentiment_status": sent_status, "extremes": extremes,
        "rating": rating_df, "rating_now": RATING_NAME[cur], "votes": votes, "plan": plan,
        "results": res, "passes": passes, "pattern_outcomes": pattern_outcomes(sig_pat, n, s),
        "allowed": allowed,
    }
