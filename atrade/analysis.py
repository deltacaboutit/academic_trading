"""Runs Frameworks 1-9 for one stock and returns everything the app displays."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import data as D
from .cleaning import clean
from .gauges import gauge_signals, gauge_values
from .gex import gex_signals
from .patterns import all_patterns
from .rating import RATING_NAME, live_votes, rating_signals, run_rating
from .sentiment import extreme_info, sentiment_signals
from .trades import pattern_trade, trade_plan
from .trend import alexander_signals, trend_signals, trend_state
from dataclasses import replace

from . import lmw
from .results import lmw_settings, load_bootstrap
from .validation import (blocks, evaluate_rules, forecast_accuracy, full_pass_table, pass_table, pattern_outcomes,
                         rating_performance, reality_check, robustness_checks, rule_daily_excess)
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


def apply_tuned_weights(s: dict) -> dict:
    """Framework 7: optionally replace the starting weights by the ones tuned on training data."""
    if not s.get("use_tuned_weights"):
        return s
    from .results import load_json
    t = (load_json("tuned_weights.json") or {}).get("tuned")
    if not t:
        return s
    s = dict(s)
    s.update({"w_trend": t["w_trend"], "w_reversal": t["w_reversal"], "w_sentiment": t["w_sentiment"],
              "buy_cutoff": t["cutoff"], "sell_cutoff": -t["cutoff"]})
    return s


def analyze(ticker: str, s: dict, refresh: bool = False, progress=None, prices=None) -> dict:
    s = apply_tuned_weights(s)

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
    sig_pat = _dedupe(all_patterns(c, s, state, df["High"].to_numpy(float), df["Low"].to_numpy(float)))
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

    step("[LMW] kernel-regression patterns (shadow ratings only)")
    lset = lmw_settings()
    lmw_det = lmw.detect(c, df["Volume"].to_numpy(float), lset["h"])
    sig_lmw = lmw.to_signals(lmw_det, c, lset["directions"], live_days=int(s["live_days_indicator"]))

    step("[GEX] add-on: gamma levels (back-test only, no rating vote)")
    gex_lv = None
    sig_gex: list = []
    try:
        if ticker.upper() != "DEMO":
            gex_lv = D.load_gex(ticker, df.index)
        if gex_lv is not None:
            sig_gex = gex_signals(df, gex_lv)
    except Exception:
        gex_lv = None

    step("Framework 9: back-testing every rule")
    by_rule = defaultdict(list)
    for sg in all_sigs + sig_alex + sig_gex + sig_lmw:
        by_rule[sg.rule].append(sg)
    res = evaluate_rules(by_rule, df, s)
    from .bootstrap import verdict_table
    boot = load_bootstrap(ticker) if ticker.upper() != "DEMO" else None
    passes = full_pass_table(res, s, verdict_table(boot, s) if boot else None)
    allowed = None
    if s.get("only_passed_rules"):
        allowed = set(passes.loc[passes["verdict"].str.startswith("Passed"), "rule"])

    step("Framework 7: rating engine")
    rating_df, index = run_rating(n, all_sigs, state, rv, s, allowed)
    rating_df.index = df.index
    rsig = rating_signals(rating_df["rating"].to_numpy())

    # [LMW] shadow ratings: B = LMW patterns instead of textbook patterns, C = both at half weight
    step("[LMW] shadow ratings B and C")
    sig_b = [x for x in all_sigs if x.kind != "pattern"] + sig_lmw
    sig_c = ([x for x in all_sigs if x.kind != "pattern"] + [replace(x, weight=x.weight * 0.5) for x in sig_pat]
             + [replace(x, weight=x.weight * 0.5) for x in sig_lmw])
    rating_b, _ = run_rating(n, sig_b, state, rv, s, None)
    rating_c, _ = run_rating(n, sig_c, state, rv, s, None)
    rating_b.index = rating_c.index = df.index
    res_rating = evaluate_rules({"Official rating (variant A)": rsig,
                                 "[LMW] Shadow rating B": rating_signals(rating_b["rating"].to_numpy()),
                                 "[LMW] Shadow rating C": rating_signals(rating_c["rating"].to_numpy())}, df, s)
    res = pd.concat([res, res_rating], ignore_index=True)

    step("Framework 9: data-snooping check, forecast accuracy, robustness")
    a_ft, b_ft = blocks(n, s)["Final test"]
    cost1 = float(s["cost_one_way"]) + float(s["spread_one_way"])
    rc_rules = {k: v for k, v in by_rule.items() if not k.startswith("Alexander")}
    rc = reality_check(rule_daily_excess(rc_rules, c, int(s["judge_period"]), 2 * cost1, a_ft, b_ft))
    fc = forecast_accuracy(sig_pat, c, a_ft, b_ft)
    fc_all = forecast_accuracy(sig_pat, c)
    robust = robustness_checks(by_rule, df, s)
    variants = pd.DataFrame([{"variant": nm, **rating_performance(rr["rating"].to_numpy(), c, a_ft, b_ft, cost1)}
                             for nm, rr in (("A - official (textbook)", rating_df), ("B - LMW patterns", rating_b),
                                            ("C - both, half weight", rating_c))])
    a_tr, b_tr = blocks(n, s)["Training"]
    lmw_info = {"Training": lmw.info_table(lmw_det, c, a_tr, b_tr), "Final test": lmw.info_table(lmw_det, c, a_ft, b_ft),
                "Whole history": lmw.info_table(lmw_det, c)}
    passes = full_pass_table(res, s, verdict_table(boot, s) if boot else None)

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
        "allowed": allowed, "gex": gex_lv, "gex_signals": sig_gex,
        "lmw_detections": lmw_det, "lmw_signals": sig_lmw, "lmw_settings": lset, "lmw_info": lmw_info,
        "rating_b": rating_b, "rating_c": rating_c, "variants": variants,
        "reality_check": rc, "forecast": {"Final test": fc, "Whole history": fc_all}, "robustness": robust,
        "bootstrap": boot, "settings_used": s,
    }
