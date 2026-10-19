"""Framework 7 - rating engine.

Group scores (each from -1 to +1):
  Trend (T)     = 0.5 x trend state + 0.5 x average of live crossover/breakout votes
  Reversal (R)  = sum of live reversal votes x their weights, capped at +/-1
                  (pattern 1.0, support bounce 1.0, resistance bounce 0.5, RSI/Bollinger/ROC 0.5)
  Sentiment (S) = sum of live sentiment votes x weights, capped at +/-1 (buy 1.0, sell 0.5)
Volume (Framework 5) multiplies each Trend and Reversal vote by 1, 0.5 or 0.

Score = 0.35 T + 0.40 R + 0.25 S.

Tool choices (the design does not say how to combine several votes):
  * Each rule has at most one live vote: its most recent signal, while it is live.
    A newer signal from the same rule replaces the older one.
  * R and S add the weighted votes and cap the total at +/-1, so the weights set
    how much one signal moves the group (one RSI signal = 0.5, one pattern = 1.0).
  * Context check: against a strong trend (trend state +1 or -1), only confirmed
    chart patterns may vote for a reversal; zone bounces and RSI/Bollinger/ROC
    votes against the trend are ignored.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .signals import Signal
from .volume import volume_weight


def _index(signals):
    by_rule = defaultdict(list)
    for sg in signals:
        by_rule[sg.rule].append(sg)
    for r in by_rule:
        by_rule[r].sort(key=lambda x: x.idx)
    return {r: (np.array([x.idx for x in v]), v) for r, v in by_rule.items()}


def live_votes(t, index, state_t, rv, s):
    """List of (signal, contribution) live on day t."""
    out = []
    use_vol = bool(s["use_volume_weights"])
    for rule, (idxs, sigs) in index.items():
        k = int(np.searchsorted(idxs, t, side="right")) - 1
        if k < 0:
            continue
        sg = sigs[k]
        if sg.live_until is not None and t > sg.live_until:
            continue
        if sg.kind == "benchmark":
            continue
        vm = volume_weight(rv[sg.idx], s) if (use_vol and sg.vol_adjust and sg.group != "sentiment") else 1.0
        if sg.group == "reversal" and sg.kind != "pattern" and state_t != 0 and sg.direction == -state_t:
            out.append((sg, 0.0, "ignored: against a strong trend (needs a chart pattern)"))
            continue
        w = 1.0 if sg.group == "trend" else sg.weight
        out.append((sg, sg.direction * w * vm, f"volume weight {vm:g}" if (use_vol and sg.vol_adjust and sg.group != "sentiment") else ""))
    return out


def run_rating(n, signals: list[Signal], state, rv, s, allowed: set | None = None) -> tuple[pd.DataFrame, dict]:
    sigs = [x for x in signals if x.kind != "benchmark" and (allowed is None or x.rule in allowed)]
    index = _index(sigs)
    T = np.zeros(n); R = np.zeros(n); S = np.zeros(n)
    for t in range(n):
        votes = live_votes(t, index, int(state[t]), rv, s)
        tv = [c for (sg, c, _) in votes if sg.group == "trend"]
        T[t] = 0.5 * state[t] + 0.5 * (np.mean(tv) if tv else 0.0)
        R[t] = np.clip(sum(c for (sg, c, _) in votes if sg.group == "reversal"), -1, 1)
        S[t] = np.clip(sum(c for (sg, c, _) in votes if sg.group == "sentiment"), -1, 1)
    wT, wR, wS = float(s["w_trend"]), float(s["w_reversal"]), float(s["w_sentiment"])
    score = wT * T + wR * R + wS * S
    G = np.vstack([T, R, S])
    pos = (G > 0).sum(axis=0); neg = (G < 0).sum(axis=0)
    veto = float(s["veto_level"])
    raw = np.zeros(n, int)
    buy = (score >= s["buy_cutoff"]) & (pos >= 2) & (G.min(axis=0) >= -veto)
    sell = (score <= s["sell_cutoff"]) & (neg >= 2) & (G.max(axis=0) <= veto)
    raw[buy] = 1; raw[sell] = -1
    # a rating changes only when the new rating holds for `confirm_days` days in a row
    k = int(s["confirm_days"])
    official = np.zeros(n, int); cur = 0; run = 0
    for t in range(n):
        if t > 0 and raw[t] == raw[t - 1]:
            run += 1
        else:
            run = 1
        if raw[t] != cur and run >= k:
            cur = raw[t]
        official[t] = cur
    df = pd.DataFrame({"T": T, "R": R, "S": S, "score": score, "raw": raw, "rating": official})
    return df, index


RATING_NAME = {1: "BUY", -1: "SELL", 0: "HOLD"}


def rating_signals(rating: np.ndarray) -> list[Signal]:
    """Turn changes of the official rating into signals so the rating itself can be back-tested."""
    out = []
    for t in range(1, len(rating)):
        if rating[t] != rating[t - 1] and rating[t] != 0:
            out.append(Signal("Official rating (variant A)", "rating", "rating", int(rating[t]), t, weight=0.0,
                              text=f"Rating changed to {RATING_NAME[int(rating[t])]}."))
        elif rating[t] != rating[t - 1] and rating[t] == 0 and rating[t - 1] != 0:
            # going to Hold closes the position: model as an 'exit' marker
            out.append(Signal("Official rating (variant A)", "rating", "exit", 0, t, weight=0.0))
    return out
