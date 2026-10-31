"""Framework 4b - chart patterns (book ch. 5 and 6).

Each detector walks through the alternating sequence of rolling-window peaks
(P) and troughs (T), as the book's HS() and FLAGS() code does, and checks the
book's conditions. A pattern only votes once price BREAKS its key line.

No look-ahead: a pattern can only be acted on once its last turning point is
confirmed (turning-point day + w). The signal day is therefore the first day,
on or after that confirmation and before the book's deadline, on which the
close is beyond the line.
"""
from __future__ import annotations

import math

import numpy as np

from .signals import Signal
from .turning_points import pips, rolling_window

TOL = 0.04


def _locals(close, w, skip_first=0):
    tp = rolling_window(close, w)
    if skip_first:
        tp = tp[tp["idx"] > skip_first]
    return (tp["idx"].to_numpy(int), tp["price"].to_numpy(float),
            tp["kind"].to_numpy(str), tp["known"].to_numpy(int))


def _line(t1, p1, t2, p2):
    s = (p2 - p1) / (t2 - t1)
    return (s, p1 - s * t1)


def _lv(line, t):
    return line[0] * t + line[1]


def _first_break(close, start, known, deadline, line, d, opposite=None):
    """First day e with max(start, known) <= e < deadline where the close is
    beyond `line` in direction d (d=-1: below, d=+1: above). If `opposite`
    (a line) is given, the pattern fails if the close crosses it first."""
    n = len(close)
    for e in range(start, min(n, int(math.ceil(deadline)))):
        c = close[e]
        if np.isnan(c):
            continue
        beyond = (c < _lv(line, e)) if d < 0 else (c > _lv(line, e))
        if opposite is not None:
            opp = (c < _lv(opposite, e)) if d > 0 else (c > _lv(opposite, e))
            if opp and not beyond:
                return None
        if beyond and e >= known:
            return e
    return None


def _trend_label(trend_state, t):
    v = trend_state[t] if 0 <= t < len(trend_state) else 0
    return {1: "uptrend", -1: "downtrend"}.get(int(v), "sideways")


# ------------------------------------------------------- head and shoulders --

def head_and_shoulders(close, s, trend_state):
    w = int(s["w_hs"])
    idx, pr, kd, kn = _locals(close, w)
    out = []
    for i in range(len(idx) - 4):
        seq = "".join(kd[i:i + 5])
        t = idx[i:i + 5]; p = pr[i:i + 5]
        if seq == "PTPTP":
            P1, T1, P2, T2, P3 = p; tP1, tT1, tP2, tT2, tP3 = t
            if not (P2 > max(P1, P3) and P1 >= 0.5 * (P3 + T2) and P3 >= 0.5 * (P1 + T1)
                    and tP2 - tP1 < 2.5 * (tP3 - tP2) and tP3 - tP2 < 2.5 * (tP2 - tP1)):
                continue
            neck = _line(tT1, T1, tT2, T2)
            e = _first_break(close, tP3 + 1, kn[i + 4], tP3 + (tP3 - tP1), neck, -1)
            if e is None:
                continue
            H = P2 - _lv(neck, tP2); W = tP3 - tP1
            prior = ""
            if i >= 2 and kd[i - 2] == "P" and kd[i - 1] == "T":
                P0, T0 = pr[i - 2], pr[i - 1]
                prior = "reversal" if (P1 > P0 and T1 > T0) else "continuation" if (P1 < P0 and T1 < T0) else ""
            out.append(Signal("Head-and-shoulders top", "reversal", "pattern", -1, e,
                              target=_lv(neck, e) - H, time_limit=e + W, line=neck,
                              points=[(a, b, l) for a, b, l in zip(t, p, ["LS", "", "Head", "", "RS"])],
                              text="Head-and-shoulders top: close broke below the neckline.",
                              extra={"height": H, "width": W, "prior": prior,
                                     "trend_before": _trend_label(trend_state, tP1)}))
        elif seq == "TPTPT":
            T1, P1, T2, P2, T3 = p; tT1, tP1, tT2, tP2, tT3 = t
            if not (T2 < min(T1, T3) and T1 <= 0.5 * (T3 + P2) and T3 <= 0.5 * (T1 + P1)
                    and tT2 - tT1 < 2.5 * (tT3 - tT2) and tT3 - tT2 < 2.5 * (tT2 - tT1)):
                continue
            neck = _line(tP1, P1, tP2, P2)
            e = _first_break(close, tT3 + 1, kn[i + 4], tT3 + (tT3 - tT1), neck, +1)
            if e is None:
                continue
            H = _lv(neck, tT2) - T2; W = tT3 - tT1
            out.append(Signal("Inverse head-and-shoulders", "reversal", "pattern", +1, e,
                              target=_lv(neck, e) + H, time_limit=e + W, line=neck,
                              points=[(a, b, l) for a, b, l in zip(t, p, ["LS", "", "Head", "", "RS"])],
                              text="Inverse head-and-shoulders: close broke above the neckline.",
                              extra={"height": H, "width": W,
                                     "trend_before": _trend_label(trend_state, tT1)}))
    return out


# ------------------------------------------------- triple and double tops --

def triple_tops_bottoms(close, s, trend_state):
    w = int(s["w_triple"])
    idx, pr, kd, kn = _locals(close, w)
    out = []
    for i in range(len(idx) - 4):
        seq = "".join(kd[i:i + 5]); t = idx[i:i + 5]; p = pr[i:i + 5]
        if seq == "PTPTP":
            P1, T1, P2, T2, P3 = p; tP1, tT1, tP2, tT2, tP3 = t
            peaks = [P1, P2, P3]
            if not ((max(peaks) - min(peaks)) / min(peaks) <= TOL and P2 <= max(P1, P3)
                    and T1 <= T2 <= (1 + TOL) * T1
                    and tP2 - tP1 < 2.5 * (tP3 - tP2) and tP3 - tP2 < 2.5 * (tP2 - tP1)):
                continue
            lvl = (0.0, T1)
            e = _first_break(close, tP3 + 1, kn[i + 4], tP3 + (tP3 - tP1), lvl, -1)
            if e is None:
                continue
            H = max(P1, P3) - T1; W = tP3 - tP1
            out.append(Signal("Triple top", "reversal", "pattern", -1, e, target=T1 - H, time_limit=e + W,
                              line=lvl, points=[(a, b, "") for a, b in zip(t, p)],
                              text="Triple top: close broke below the support at the first trough.",
                              extra={"height": H, "width": W, "trend_before": _trend_label(trend_state, tP1)}))
        elif seq == "TPTPT":
            T1, P1, T2, P2, T3 = p; tT1, tP1, tT2, tP2, tT3 = t
            tr = [T1, T2, T3]
            if not ((max(tr) - min(tr)) / min(tr) <= TOL and T2 >= min(T1, T3)
                    and P2 <= P1 <= (1 + TOL) * P2
                    and tT2 - tT1 < 2.5 * (tT3 - tT2) and tT3 - tT2 < 2.5 * (tT2 - tT1)):
                continue
            lvl = (0.0, P1)
            e = _first_break(close, tT3 + 1, kn[i + 4], tT3 + (tT3 - tT1), lvl, +1)
            if e is None:
                continue
            H = P1 - min(T1, T3); W = tT3 - tT1
            out.append(Signal("Triple bottom", "reversal", "pattern", +1, e, target=P1 + H, time_limit=e + W,
                              line=lvl, points=[(a, b, "") for a, b in zip(t, p)],
                              text="Triple bottom: close broke above the resistance at the first peak.",
                              extra={"height": H, "width": W, "trend_before": _trend_label(trend_state, tT1)}))
    return out


def double_tops_bottoms(close, s, trend_state):
    w = int(s["w_double"])
    idx, pr, kd, kn = _locals(close, w)
    out = []
    for i in range(len(idx) - 2):
        seq = "".join(kd[i:i + 3]); t = idx[i:i + 3]; p = pr[i:i + 3]
        if seq == "PTP":
            P1, T1, P2 = p; tP1, tT1, tP2 = t
            if not (abs(P1 - P2) / min(P1, P2) <= TOL and (T1 - P1) / P1 <= -0.10):
                continue
            lvl = (0.0, T1)
            e = _first_break(close, tP2 + 1, kn[i + 2], tP2 + (tP2 - tP1), lvl, -1)
            if e is None:
                continue
            H = max(P1, P2) - T1; W = tP2 - tP1
            out.append(Signal("Double top", "reversal", "pattern", -1, e, target=T1 - H, time_limit=e + W,
                              line=lvl, points=[(a, b, "") for a, b in zip(t, p)],
                              text="Double top: close broke below the trough between the two peaks.",
                              extra={"height": H, "width": W, "trend_before": _trend_label(trend_state, tP1)}))
        elif seq == "TPT":
            T1, P1, T2 = p; tT1, tP1, tT2 = t
            if not (abs(T1 - T2) / min(T1, T2) <= TOL and (P1 - T1) / T1 >= 0.10):
                continue
            lvl = (0.0, P1)
            e = _first_break(close, tT2 + 1, kn[i + 2], tT2 + (tT2 - tT1), lvl, +1)
            if e is None:
                continue
            H = P1 - min(T1, T2); W = tT2 - tT1
            out.append(Signal("Double bottom", "reversal", "pattern", +1, e, target=P1 + H, time_limit=e + W,
                              line=lvl, points=[(a, b, "") for a, b in zip(t, p)],
                              text="Double bottom: close broke above the peak between the two troughs.",
                              extra={"height": H, "width": W, "trend_before": _trend_label(trend_state, tT1)}))
    return out


# -------------------------------------------- flags, pennants and wedges --

def _prior_points(close, t_first, bull, thor):
    """Three turning points before the pattern, found with PIPs (book FLAGS()).

    The prior move starts at the lowest (bull) / highest (bear) close of the
    `thor` days before the pattern's first point. Returns [(idx, price)] x 3
    = start, X0, Y0.
    """
    a = max(0, t_first - thor)
    seg = close[a:t_first + 1]
    if len(seg) < 2 or np.isnan(seg).any():
        return None
    j = int(np.argmin(seg) if bull else np.argmax(seg))
    start = a + j
    if t_first - start + 1 < 4:
        start = max(0, start - 4)
    y = close[start:t_first + 1]
    pos = pips(y, 4, "perpendicular")
    if len(pos) < 4:
        return None
    return [(start + q, float(y[q])) for q in pos[:3]]


def _apex(upper, lower):
    if abs(upper[0] - lower[0]) < 1e-12:
        return math.inf
    return (lower[1] - upper[1]) / (upper[0] - lower[0])


def flags_pennants_wedges(close, s, trend_state):
    out = []
    thor = int(s["flag_prior_days"])
    for name, w in (("flag", int(s["w_flag"])), ("pennant", int(s["w_pennant"])), ("wedge", int(s["w_wedge"]))):
        idx, pr, kd, kn = _locals(close, w, skip_first=20)
        for i in range(len(idx) - 3):
            seq = "".join(kd[i:i + 4]); t = idx[i:i + 4]; p = pr[i:i + 4]
            if seq == "PTPT":
                sig = _bull(name, close, t, p, kn[i + 3], thor, trend_state)
            elif seq == "TPTP":
                sig = _bear(name, close, t, p, kn[i + 3], thor, trend_state)
            else:
                sig = None
            if sig is not None:
                out.append(sig)
    return out


def _bull(name, close, t, p, known, thor, trend_state):
    P1, T1, P2, T2 = p; tP1, tT1, tP2, tT2 = t
    # cheap shape checks first (the PIP search below is the slow part)
    sP_, sT_ = (P2 - P1) / (tP2 - tP1), (T2 - T1) / (tT2 - tT1)
    if name == "flag" and not (P2 < P1 and T2 < T1):
        return None
    if name == "pennant" and not (P2 < P1 and T2 > T1):
        return None
    if name == "wedge" and not (0 > sT_ > sP_):
        return None
    prior = _prior_points(close, tP1, True, thor)
    if prior is None:
        return None
    (tS, S), (tX, X), (tY, Y) = prior          # start trough, P0, T0
    upper = _line(tP1, P1, tP2, P2); lower = _line(tT1, T1, tT2, T2)
    ts = _apex(upper, lower)
    mx_val, mx_t = (X, tX) if X >= Y else (Y, tY)
    if name == "flag":
        if not (P1 > max(X, Y) and mx_val < _lv(lower, mx_t) and P2 < P1 and T2 < T1
                and (math.isinf(ts) or abs(ts - (tP1 + tT2) / 2) > 5 * (tT2 - tP1))):
            return None
        deadline = min(tT2 + (tT2 - tP1), tP1 + 20 + 1)
        e = _first_break(close, tT2 + 1, known, deadline, upper, +1, opposite=lower)
        if e is None or not (tP1 - tS > 2 * (e - tP1)):
            return None
        return _mk("Bull flag", +1, e, _lv(upper, e) + (P1 - S), e + (tP1 - tS), upper, t, p, prior,
                   "Bull flag: price broke above the flag's upper line.", trend_state)
    if name == "pennant":
        if not (min(P1, T1) > max(X, Y) and P2 < P1 and T2 > T1 and not math.isinf(ts)
                and ts > tT2 and ts - tP1 < 2 * (tT2 - tP1)):
            return None
        span = ts - tP1
        deadline = min(tP1 + 0.75 * span, tP1 + 20 + 1)
        e = _first_break(close, tT2 + 1, known, deadline, upper, +1, opposite=lower)
        if e is None:
            return None
        L = e - tP1
        if not (0.5 * span < L < 0.75 * span and L <= 20 and tP1 - tS > 2 * L):
            return None
        return _mk("Bull pennant", +1, e, _lv(upper, e) + (P1 - S), e + (tP1 - tS), upper, t, p, prior,
                   "Bull pennant: price broke above the pennant's upper line.", trend_state)
    # falling (bullish) wedge
    sP, sT = upper[0], lower[0]
    if not (T2 > max(X, Y) and 0 > sT > sP and not math.isinf(ts) and ts > tT2
            and ts - tP1 < 2 * (tT2 - tP1)):
        return None
    span = ts - tP1
    deadline = min(tP1 + 0.8 * span, tP1 + 80 + 1)
    e = _first_break(close, tT2 + 1, known, deadline, upper, +1, opposite=lower)
    if e is None:
        return None
    L = e - tP1
    if not (20 <= L <= 80 and 0.5 * span < L < 0.8 * span):
        return None
    return _mk("Falling wedge", +1, e, P1, 2 * e - tP1, upper, t, p, prior,
               "Falling wedge: price broke above the wedge's upper line.", trend_state)


def _bear(name, close, t, p, known, thor, trend_state):
    T1, P1, T2, P2 = p; tT1, tP1, tT2, tP2 = t
    sP_, sT_ = (P2 - P1) / (tP2 - tP1), (T2 - T1) / (tT2 - tT1)
    if name == "flag" and not (P2 > P1 and T2 > T1):
        return None
    if name == "pennant" and not (P2 < P1 and T2 > T1):
        return None
    if name == "wedge" and not (sT_ > sP_ > 0):
        return None
    prior = _prior_points(close, tT1, False, thor)
    if prior is None:
        return None
    (tS, S), (tX, X), (tY, Y) = prior          # start peak, T0, P0
    upper = _line(tP1, P1, tP2, P2); lower = _line(tT1, T1, tT2, T2)
    ts = _apex(upper, lower)
    mn_val, mn_t = (X, tX) if X <= Y else (Y, tY)
    if name == "flag":
        if not (T1 < min(X, Y) and mn_val > _lv(upper, mn_t) and P2 > P1 and T2 > T1
                and (math.isinf(ts) or abs(ts - (tT1 + tP2) / 2) > 5 * (tP2 - tT1))):
            return None
        deadline = min(tP2 + (tP2 - tT1), tT1 + 20 + 1)
        e = _first_break(close, tP2 + 1, known, deadline, lower, -1, opposite=upper)
        if e is None or not (tT1 - tS > 2 * (e - tT1)):
            return None
        return _mk("Bear flag", -1, e, _lv(lower, e) - (S - T1), e + (tT1 - tS), lower, t, p, prior,
                   "Bear flag: price broke below the flag's lower line.", trend_state)
    if name == "pennant":
        if not (max(T1, P1) < min(X, Y) and P2 < P1 and T2 > T1 and not math.isinf(ts)
                and ts > tP2 and ts - tT1 < 2 * (tP2 - tT1)):
            return None
        span = ts - tT1
        deadline = min(tT1 + 0.75 * span, tT1 + 20 + 1)
        e = _first_break(close, tP2 + 1, known, deadline, lower, -1, opposite=upper)
        if e is None:
            return None
        L = e - tT1
        if not (0.5 * span < L < 0.75 * span and L <= 20 and tT1 - tS > 2 * L):
            return None
        return _mk("Bear pennant", -1, e, _lv(lower, e) - (S - T1), e + (tT1 - tS), lower, t, p, prior,
                   "Bear pennant: price broke below the pennant's lower line.", trend_state)
    sP, sT = upper[0], lower[0]
    if not (P2 < min(X, Y) and sT > sP > 0 and not math.isinf(ts) and ts > tP2
            and ts - tT1 < 2 * (tP2 - tT1)):
        return None
    span = ts - tT1
    deadline = min(tT1 + 0.8 * span, tT1 + 80 + 1)
    e = _first_break(close, tP2 + 1, known, deadline, lower, -1, opposite=upper)
    if e is None:
        return None
    L = e - tT1
    if not (20 <= L <= 80 and 0.5 * span < L < 0.8 * span):
        return None
    return _mk("Rising wedge", -1, e, T1, 2 * e - tT1, lower, t, p, prior,
               "Rising wedge: price broke below the wedge's lower line.", trend_state)


def _mk(rule, d, e, target, tl, line, t, p, prior, text, trend_state):
    pts = [(a, b, "prior") for a, b in prior] + [(a, b, "") for a, b in zip(t, p)]
    return Signal(rule, "reversal", "pattern", d, int(e), target=float(target), time_limit=int(tl),
                  line=line, points=pts, text=text,
                  extra={"trend_before": _trend_label(trend_state, int(t[0]))})


# ---------------------------------------------------- rounding (circle) ----

def _circle_fit(y_seg, bounds):
    """Book's RBcircles(): scale the stretch so depth = width/2, build the circle
    through A, B, C and count the share of closes inside the band."""
    n = len(y_seg)
    w1 = n
    lo, hi = y_seg.min(), y_seg.max()
    if hi - lo <= 0:
        return 0.0
    ys = 1 + (y_seg - lo) * (w1 / 2 - 1) / (hi - lo)        # mapminmax(.., 1, w1/2)
    xs = np.arange(n, dtype=float)
    Ax, Cx = 0.0, float(n - 1)
    Ay = (ys[0] + ys[-1]) / 2
    Bx, By = (Ax + Cx) / 2, 1.0
    a = math.hypot(Bx - Cx, By - Ay)
    AB = math.hypot(Ax - Bx, Ay - By)
    if AB == 0 or Ay - By <= 0:
        return 0.0
    sinA = (Ay - By) / AB
    R = a / (2 * sinA)
    Kx, Ky = Bx, By + R

    def circle_y(r):
        disc = r * r - (xs - Kx) ** 2
        yv = Ky - np.sqrt(np.clip(disc, 0, None))
        yv[disc < 0] = Ky
        return yv

    inner = circle_y(R * (1 - bounds))   # higher curve
    outer = circle_y(R * (1 + bounds))   # lower curve
    inside = (ys <= inner) & (ys >= outer)
    return float(inside.mean())


def rounding(close, s, trend_state):
    w = int(s["w_round"]); b = float(s["round_bounds"])
    min_w = int(s["round_min_width"]); tfit = float(s["round_fit"])
    out = []
    n = len(close)
    tp = rolling_window(close, w)
    for bottom in (True, False):
        y = close if bottom else -close
        sel = tp[tp["kind"] == ("P" if bottom else "T")]
        for t1 in sel["idx"].to_numpy(int):
            lvl = y[t1]
            later = np.where(y[t1 + 1:] >= lvl)[0]
            if len(later) == 0:
                continue
            t2 = t1 + 1 + int(later[0])
            width = t2 - t1 + 1
            if width < min_w or t1 + w > t2:
                continue
            seg = y[t1:t2 + 1]
            if np.isnan(seg).any():
                continue
            fit = _circle_fit(seg, b)
            if fit < tfit:
                continue
            tT = t1 + int(np.argmin(seg))
            ext = float(close[tT])
            if bottom:
                depth_px = max(close[t1], close[t2]) - ext
                target = close[t1] + depth_px; d = +1; rule = "Rounding bottom"
                text = "Rounding bottom (saucer): close rose back above the left rim."
            else:
                depth_px = ext - min(close[t1], close[t2])
                target = close[t1] - depth_px; d = -1; rule = "Rounding top"
                text = "Rounding top: close fell back below the left rim."
            out.append(Signal(rule, "reversal", "pattern", d, int(t2), target=float(target),
                              time_limit=int(t2 + width), line=(0.0, float(close[t1])), vol_adjust=False,
                              points=[(t1, float(close[t1]), "rim"), (tT, ext, "low" if bottom else "high"),
                                      (t2, float(close[t2]), "break")],
                              text=text,
                              extra={"fit": fit, "width": width, "depth": depth_px / (ext if bottom else min(close[t1], close[t2])),
                                     "trend_before": _trend_label(trend_state, t1)}))
    return out


# ===================================================================
# Patterns added from PatternFinder 2.5. They are not in the book, so
# each rule names its own source; "tool choice" marks our own choices.
# They follow the same no-look-ahead rule as the patterns above.
# ===================================================================

# ------------------------------------------------ triangles and rectangle --

def _first_break_either(close, start, known, deadline, upper, lower):
    """First day e with max(start, known) <= e < deadline on which the close is
    above `upper` (returns e, +1) or below `lower` (returns e, -1)."""
    for e in range(max(start, known), min(len(close), int(math.ceil(deadline)))):
        c = close[e]
        if np.isnan(c):
            continue
        if c > _lv(upper, e):
            return e, +1
        if c < _lv(lower, e):
            return e, -1
    return None, 0


def _flat(v):
    return (max(v) - min(v)) / min(v) <= TOL


def triangles_rectangles(close, s, trend_state):
    """Ascending, descending and symmetrical triangles and rectangles.

    Shape: five alternating turning points, as in Lo, Mamaysky & Wang (2000)
    (their triangle and rectangle use E1..E5). A side is "flat" when its points
    are within TOL (4%) of each other, the same tolerance as triple tops.
      Ascending triangle   flat tops, rising lows        -> breaks up (Edwards & Magee)
      Descending triangle  flat lows, falling tops       -> breaks down (Edwards & Magee)
      Symmetrical triangle falling tops and rising lows  -> breaks either way
      Rectangle            flat tops and flat lows       -> breaks either way
    Triangles must break between half and three-quarters of the way from the
    first point to the apex, where the two lines meet (Edwards & Magee).
    Rectangles must break within one pattern width of the last point (as for
    head-and-shoulders). Target: the pattern's height at its widest part,
    measured from the broken line (Edwards & Magee; Murphy).
    """
    out, seen = [], set()
    for kinds, w in ((("asc", "desc", "sym"), int(s["w_triangle"])), (("rect",), int(s["w_rect"]))):
        idx, pr, kd, kn = _locals(close, w)
        for i in range(len(idx) - 4):
            seq = "".join(kd[i:i + 5])
            if seq not in ("PTPTP", "TPTPT"):
                continue
            t = idx[i:i + 5]; p = pr[i:i + 5]
            hi = [(a, b) for a, b, k in zip(t, p, seq) if k == "P"]
            lo = [(a, b) for a, b, k in zip(t, p, seq) if k == "T"]
            hv = [b for _, b in hi]; lv = [b for _, b in lo]
            falling = all(x > y for x, y in zip(hv, hv[1:]))
            rising = all(x < y for x, y in zip(lv, lv[1:]))
            fh, fl = _flat(hv), _flat(lv)
            if fh and fl:
                # the band must be taller than the flatness tolerance, or tops and bottoms blur (tool choice)
                kind = "rect" if min(hv) > max(lv) and np.mean(hv) > (1 + TOL) * np.mean(lv) else None
            elif fh and rising:
                kind = "asc"
            elif fl and falling:
                kind = "desc"
            elif falling and rising:
                kind = "sym"
            else:
                kind = None
            if kind not in kinds:
                continue
            t1, t5, known = int(t[0]), int(t[-1]), int(kn[i + 4])
            W = t5 - t1
            sloped_up = _line(hi[0][0], hi[0][1], hi[-1][0], hi[-1][1])
            sloped_lo = _line(lo[0][0], lo[0][1], lo[-1][0], lo[-1][1])
            upper = (0.0, max(hv)) if kind in ("asc", "rect") else sloped_up
            lower = (0.0, min(lv)) if kind in ("desc", "rect") else sloped_lo
            H = _lv(upper, t1) - _lv(lower, t1)
            if H <= 0:
                continue
            if kind == "rect":
                e, d = _first_break_either(close, t5 + 1, known, t5 + W, upper, lower)
            else:
                ts = _apex(upper, lower)
                if math.isinf(ts) or ts <= t5:
                    continue
                span = ts - t1
                deadline = t1 + 0.75 * span
                if kind == "sym":
                    e, d = _first_break_either(close, t5 + 1, known, deadline, upper, lower)
                elif kind == "asc":
                    d = +1; e = _first_break(close, t5 + 1, known, deadline, upper, +1, opposite=lower)
                else:
                    d = -1; e = _first_break(close, t5 + 1, known, deadline, lower, -1, opposite=upper)
                if e is None or not e - t1 > 0.5 * span:
                    continue
            rule = {"asc": "Ascending triangle", "desc": "Descending triangle",
                    "sym": "Symmetrical triangle", "rect": "Rectangle"}[kind]
            if e is None or (rule, e, d) in seen:
                continue                    # two overlapping five-point windows can find the same breakout
            seen.add((rule, e, d))
            line = upper if d > 0 else lower
            where = "above the upper line" if d > 0 else "below the lower line"
            out.append(Signal(rule, "reversal", "pattern", d, int(e), target=float(_lv(line, e) + d * H),
                              time_limit=int(e + W), line=line, points=[(a, b, "") for a, b in zip(t, p)],
                              text=f"{rule}: close broke {where}.",
                              extra={"height": H, "width": W, "trend_before": _trend_label(trend_state, t1)}))
    return out


# --------------------------------------------------------- cup and handle --

def cup_and_handle(close, s, trend_state):
    """Cup and handle (O'Neil). Bullish only.

    Two peaks (the rims) within TOL (4%) of each other, 7 to 65 weeks apart, with
    no close between them above the lower rim. The cup is 12-50% deep, measured
    from the left rim (O'Neil), with its low in the middle 60% of the cup (tool
    choice, as in PatternFinder 2.5). It must be round, not V-shaped (O'Neil):
    it must pass the book's circle test used for rounding bottoms, and at least
    half of its closes must lie in the lowest 40% of its depth (tool choice: a
    round cup spends about 60% of its time there, a straight-sided V only 40%).
    After the right rim comes the handle, a small dip that stays in the cup's
    upper half (O'Neil) and lasts at most half the cup's width and 40 days (tool
    choice, as in PatternFinder 2.5). The signal is the first close above the right rim once
    that rim is confirmed, which makes the handle at least w_cup days long.
    Target: the rim plus the cup's depth (Bulkowski's measure rule).
    """
    w = int(s["w_cup"]); b = float(s["round_bounds"]); tfit = float(s["round_fit"])
    mn_w, mx_w = int(s["cup_min_days"]), int(s["cup_max_days"])
    mn_d, mx_d = float(s["cup_min_depth"]), float(s["cup_max_depth"])
    tp = rolling_window(close, w)
    pk = tp[tp["kind"] == "P"]
    pt, pp, pkn = pk["idx"].to_numpy(int), pk["price"].to_numpy(float), pk["known"].to_numpy(int)
    best = {}
    for a in range(len(pt)):
        tL, PL = pt[a], pp[a]
        for c in range(a + 1, len(pt)):
            tR, PR = pt[c], pp[c]
            width = int(tR - tL)
            if width > mx_w:
                break
            inside = close[tL + 1:tR]
            if np.isnan(inside).any() or inside.max() > PL:
                break                       # a close above the left rim: no wider cup from this rim either
            if PR > (1 + TOL) * PL:
                break                       # this peak sits inside every wider cup
            if width < mn_w or abs(PR - PL) / min(PL, PR) > TOL or inside.max() > PR:
                continue
            tB = tL + int(np.argmin(close[tL:tR + 1])); B = float(close[tB])
            depth = min(PL, PR) - B
            if not (mn_d <= (PL - B) / PL <= mx_d and 0.2 <= (tB - tL) / width <= 0.8):
                continue
            seg = close[tL:tR + 1]
            if np.mean(seg <= B + 0.4 * depth) < 0.5:
                continue
            fit = _circle_fit(seg, b)
            if fit < tfit:
                continue
            max_handle = max(w, min(40, width // 2))
            e = _first_break(close, tR + 1, int(pkn[c]), tR + max_handle + 1, (0.0, PR), +1,
                             opposite=(0.0, PR - 0.5 * depth))
            if e is None:
                continue
            tH = tR + 1 + int(np.argmin(close[tR + 1:e]))
            sig = Signal("Cup and handle", "reversal", "pattern", +1, int(e), target=float(PR + depth),
                         time_limit=int(e + width), line=(0.0, float(PR)),
                         points=[(tL, float(PL), "left rim"), (tB, B, "bottom"), (tR, float(PR), "right rim"),
                                 (tH, float(close[tH]), "handle"), (int(e), float(close[e]), "break")],
                         text="Cup and handle: close broke above the right rim after the handle.",
                         extra={"fit": fit, "width": width, "depth": depth / PL,
                                "trend_before": _trend_label(trend_state, int(tL))})
            if e not in best or fit > best[e].extra["fit"]:
                best[e] = sig
    return [best[e] for e in sorted(best)]


# -------------------------------------------------------- island reversals --

def islands(high, low, close, s, trend_state):
    """Island top and island bottom (Murphy, Charting Made Easy).

    Island top: a gap up (today's low above yesterday's high) to a new
    `island_prior_days`-day closing high, so the gap ends a rise (tool choice).
    Then, within `island_max_days` days, a gap down below every low of the
    island days, which leaves them stranded. The signal day is the day of the
    second gap. If the island days fall back into the first gap first, there is
    no island. Island bottom: the same, upside down.
    Line: the island's edge; closing back across it means the second gap
    has been filled. Target: the island's own height, measured from that edge
    (tool choice, as in PatternFinder 2.5; Murphy gives no target).
    Time limit: `default_hold` days (Murphy gives none).
    Needs daily highs and lows, so it is skipped when only closes are known.
    """
    out = []
    m, k = int(s["island_max_days"]), int(s["island_prior_days"])
    hold = int(s["default_hold"])
    for top in (True, False):
        # an island bottom is an island top on the upside-down chart
        H, L, C = (high, low, close) if top else (-low, -high, -close)
        f = 1.0 if top else -1.0
        n = len(C)
        for a in range(k, n - 1):
            if not (L[a] > H[a - 1] and C[a] > np.max(C[a - k:a])):
                continue
            for e in range(a, min(n - 1, a + m)):
                isl_lo = np.min(L[a:e + 1])
                if np.isnan(isl_lo) or isl_lo <= H[a - 1]:
                    break                     # fell back into the first gap: no island
                if H[e + 1] < isl_lo:         # second gap: the island is complete
                    j = a + int(np.argmax(H[a:e + 1])); isl_hi = float(H[j])
                    height = isl_hi - isl_lo
                    rule = "Island top" if top else "Island bottom"
                    out.append(Signal(rule, "reversal", "pattern", -1 if top else +1, int(e + 1),
                                      target=float(f * (isl_lo - height)), time_limit=int(e + 1 + hold),
                                      line=(0.0, float(f * isl_lo)),
                                      points=[(a - 1, float(f * H[a - 1]), "gap"), (j, float(f * isl_hi), "island"),
                                              (e + 1, float(f * H[e + 1]), "gap")],
                                      text=f"{rule}: price gapped {'down' if top else 'up'} and left the island behind.",
                                      extra={"island_days": e - a + 1,
                                             "trend_before": _trend_label(trend_state, a - 1)}))
                    break
    return out


def all_patterns(close, s, trend_state, high=None, low=None) -> list[Signal]:
    """Every chart pattern. Island reversals need daily highs and lows; without
    them (the random-price tests use closes only) they are left out."""
    close = np.asarray(close, float)
    sigs = []
    sigs += head_and_shoulders(close, s, trend_state)
    sigs += triple_tops_bottoms(close, s, trend_state)
    sigs += double_tops_bottoms(close, s, trend_state)
    sigs += flags_pennants_wedges(close, s, trend_state)
    sigs += rounding(close, s, trend_state)
    sigs += triangles_rectangles(close, s, trend_state)
    sigs += cup_and_handle(close, s, trend_state)
    if high is not None and low is not None:
        sigs += islands(np.asarray(high, float), np.asarray(low, float), close, s, trend_state)
    for g in sigs:
        g.weight = float(s["wt_pattern"])
    return sigs
