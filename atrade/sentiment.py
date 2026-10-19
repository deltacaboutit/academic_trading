"""Framework 6 - contrarian options sentiment (Summa, Trading Against the Crowd).

A HIGH oscillator (many puts / expensive options) means the crowd is fearful,
which is a BUY setup; a LOW oscillator is a SELL setup.
"""
from __future__ import annotations

import numpy as np

from .indicators import ema
from .signals import Signal


def oscillator(series: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Osc(t) = 100 * (EMA_fast(S)_{t-1} - EMA_slow(S)_{t-1}) / EMA_slow(S)_{t-1}."""
    ef, es = ema(series, fast), ema(series, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = 100 * (ef - es) / es
    out = np.full(len(o), np.nan)
    out[1:] = o[:-1]
    return out


def _valid(osc, t):
    return not np.isnan(osc[t])


# ------------------------------------------------------------ systems -------

def squeeze_play_1(osc5, osc21, high, low, close, name, s):
    n = len(close); pos = 0; sigs = []; cur = None
    for t in range(1, n):
        if not (_valid(osc5, t) and _valid(osc5, t - 1)):
            continue
        # exits on the 21/50 oscillator crossing zero
        if pos != 0 and _valid(osc21, t) and _valid(osc21, t - 1):
            if pos == 1 and osc21[t - 1] > 0 and osc21[t] < 0:
                cur.live_until = t; pos = 0; cur = None
            elif pos == -1 and osc21[t - 1] < 0 and osc21[t] > 0:
                cur.live_until = t; pos = 0; cur = None
        long_e = osc5[t - 1] > 0 and osc5[t] < 0 and close[t] > high[t - 1]
        short_e = osc5[t - 1] < 0 and osc5[t] > 0 and close[t] < low[t - 1]
        for d, ok in ((1, long_e), (-1, short_e)):
            if ok and pos != d:
                if cur is not None:
                    cur.live_until = t
                cur = Signal(name, "sentiment", "sentiment", d, t,
                             weight=s["wt_sent_buy"] if d > 0 else s["wt_sent_sell"], vol_adjust=False,
                             text=("Fear in the equity put/call ratio started to ease and price confirmed."
                                   if d > 0 else "Complacency in the equity put/call ratio started to fade and price confirmed."))
                sigs.append(cur); pos = d          # live until its exit (None = still open)
    return sigs


def squeeze_play_2a(osc, open_, high, low, close, name, s):
    n = len(close); lb = int(s["sp2_lookback"]); X = float(s["sp2_extreme"])
    pos = 0; sigs = []; cur = None; entry = None; peak = 0.0; start = 0
    for t in range(max(lb, 3), n):
        if not _valid(osc, t):
            continue
        win = osc[t - lb + 1:t + 1]
        if np.isnan(win).any():
            continue
        if pos != 0 and t > start + 1:
            if entry:
                prof = pos * (close[t] / entry - 1)
                peak = max(peak, prof)
                exit_ = False
                if (pos == 1 and osc[t] < -X) or (pos == -1 and osc[t] > X):
                    exit_ = True                                   # opposite extreme
                if prof <= -float(s["sp2a_max_loss"]):
                    exit_ = True                                   # 10% max loss
                if t - start > int(s["sp2a_trail_delay"]) and peak > 0 and prof <= peak * (1 - float(s["sp2a_trail_giveback"])):
                    exit_ = True                                   # trailing stop
                if exit_:
                    cur.live_until = t; pos = 0; cur = None
                    continue
        hh = np.max(high[t - 3:t]); ll = np.min(low[t - 3:t])
        long_e = win.max() > X and close[t] > hh
        short_e = win.min() < -X and close[t] < ll
        for d, ok in ((1, long_e), (-1, short_e)):
            if ok and pos != d:
                if cur is not None:
                    cur.live_until = t
                cur = Signal(name, "sentiment", "sentiment", d, t,
                             weight=s["wt_sent_buy"] if d > 0 else s["wt_sent_sell"], vol_adjust=False,
                             text=("A recent fear extreme in the stock's put/call ratio, confirmed by a close above the 3-day high."
                                   if d > 0 else "A recent greed extreme in the stock's put/call ratio, confirmed by a close below the 3-day low."))
                sigs.append(cur); pos = d; peak = 0.0; start = t
                entry = open_[t + 1] if t + 1 < n else None
                break
    return sigs


def squeeze_play_2b(osc, high, low, close, trigger, hold, name, label, s):
    n = len(close); lb = int(s["sp2_lookback"]); X = float(s["sp2_extreme"])
    sigs = []; busy_until = -1
    for t in range(max(lb, trigger), n):
        if t <= busy_until:
            continue
        win = osc[t - lb + 1:t + 1]
        if np.isnan(win).any():
            continue
        if trigger > 0:
            hh = np.max(high[t - trigger:t]); ll = np.min(low[t - trigger:t])
        else:
            hh, ll = -np.inf, np.inf
        for d, ok in ((1, win.max() > X and close[t] > hh), (-1, win.min() < -X and close[t] < ll)):
            if ok:
                sigs.append(Signal(name, "sentiment", "sentiment", d, t, live_until=t + hold,
                                   weight=s["wt_sent_buy"] if d > 0 else s["wt_sent_sell"], vol_adjust=False,
                                   text=f"Recent {'fear' if d > 0 else 'complacency'} extreme in {label}, "
                                        f"confirmed by price; held {hold} days."))
                busy_until = t + hold
                break
    return sigs


def tsunami(osc, n, hold, s, name):
    X = float(s["tsunami_level"]); sigs = []; busy_until = -1
    for t in range(n):
        if t <= busy_until or np.isnan(osc[t]):
            continue
        d = 1 if osc[t] > X else -1 if osc[t] < -X else 0
        if d:
            sigs.append(Signal(name, "sentiment", "sentiment", d, t, live_until=t + hold,
                               weight=s["wt_sent_buy"] if d > 0 else s["wt_sent_sell"], vol_adjust=False,
                               text=f"Tsunami Sentiment Wave: the stock's 21/50 put/call oscillator is "
                                    f"{'above +' if d > 0 else 'below -'}{X:.0f}%."))
            busy_until = t + hold
    return sigs


def fast_vol_swing(osc, n, s, name):
    X = float(s["fastvol_level"]); sigs = []; pos = 0; cur = None
    for t in range(1, n):
        if np.isnan(osc[t]) or np.isnan(osc[t - 1]):
            continue
        if pos == 1 and osc[t - 1] > 0 and osc[t] <= 0 or pos == -1 and osc[t - 1] < 0 and osc[t] >= 0:
            cur.live_until = t; pos = 0; cur = None
            continue
        if pos == 0:
            d = 1 if osc[t] > X else -1 if osc[t] < -X else 0
            if d:
                cur = Signal(name, "sentiment", "sentiment", d, t, live_until=None,
                             weight=s["wt_sent_buy"] if d > 0 else s["wt_sent_sell"], vol_adjust=False,
                             text=f"Fast volatility swing: VIX 1/10 oscillator {'above +' if d > 0 else 'below -'}{X:.0f}%.")
                sigs.append(cur); pos = d
    return sigs


# ---------------------------------------------------------- orchestration ---

def extreme_info(series: np.ndarray) -> dict | None:
    """Very slow (10/250) oscillator vs. its record so far (threshold studies)."""
    o = oscillator(series, 10, 250)
    valid = ~np.isnan(o)
    if valid.sum() < 20:
        return None
    hist = o[valid]
    cur = hist[-1]; hi, lo = hist.max(), hist.min()
    rng = hi - lo if hi > lo else 1.0
    return {"value": float(cur), "record_high": float(hi), "record_low": float(lo),
            "near_high": bool(cur >= hi - 0.10 * rng), "near_low": bool(cur <= lo + 0.10 * rng)}


def sentiment_signals(series: dict, df, s: dict):
    """series: dict name -> numpy array aligned to df's dates (NaN where missing).

    Returns (signals, status) where status explains which systems ran.
    """
    o = df["Open"].to_numpy(float); h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float); c = df["Close"].to_numpy(float)
    n = len(c)
    sigs: list[Signal] = []
    status: list[dict] = []

    def have(name, need):
        arr = series.get(name)
        cnt = 0 if arr is None else int((~np.isnan(arr)).sum())
        return arr is not None and cnt >= need, cnt

    # Squeeze Play I on the equity-only put/call ratio
    ok, cnt = have("equity_pc", 60)
    if ok:
        pc = series["equity_pc"]
        sigs += squeeze_play_1(oscillator(pc, 5, 21), oscillator(pc, 21, 50), h, l, c, "Squeeze Play I (equity P/C)", s)
    status.append({"system": "Squeeze Play I", "data": "Equity-only put/call ratio", "days": cnt, "needed": 60, "active": ok})

    # Stock's own put/call ratio: Squeeze Play II (A and B) and Tsunami
    ok, cnt = have("stock_pc", 110)
    if ok:
        pc = series["stock_pc"]
        o50 = oscillator(pc, 50, 100)
        sigs += squeeze_play_2a(o50, o, h, l, c, "Squeeze Play II-A (stock P/C)", s)
        sigs += squeeze_play_2b(o50, h, l, c, 3, int(s["sp2b_hold"]), "Squeeze Play II-B (stock P/C)", "the stock's put/call ratio", s)
    status.append({"system": "Squeeze Play II-A and II-B", "data": "Stock's own put/call ratio", "days": cnt, "needed": 110, "active": ok})
    ok2, cnt2 = have("stock_pc", 60)
    if ok2:
        sigs += tsunami(oscillator(series["stock_pc"], 21, 50), n, int(s["tsunami_hold"]), s, "Tsunami (stock P/C)")
    status.append({"system": "Tsunami Sentiment Wave", "data": "Stock's own put/call ratio", "days": cnt2, "needed": 60, "active": ok2})

    # Stock's own implied volatility: Squeeze Play II-B, 21/50, 1-day trigger
    ok, cnt = have("stock_iv", 60)
    if ok:
        sigs += squeeze_play_2b(oscillator(series["stock_iv"], 21, 50), h, l, c, 1, int(s["sp2b_hold"]),
                                "Squeeze Play II-B (stock IV)", "the stock's implied volatility", s)
    status.append({"system": "Squeeze Play II-B on implied volatility", "data": "Stock's own implied volatility", "days": cnt, "needed": 60, "active": ok})

    # VIX: Squeeze Play II-B and the fast volatility swing
    f, sl = s["vix_osc"]
    ok, cnt = have("vix", int(sl) + 10)
    if ok:
        vx = series["vix"]
        sigs += squeeze_play_2b(oscillator(vx, int(f), int(sl)), h, l, c, int(s["vix_trigger_days"]), int(s["sp2b_hold"]),
                                f"Squeeze Play II-B (VIX {f}/{sl})", "the VIX", s)
        sigs += fast_vol_swing(oscillator(vx, 1, 10), n, s, "Fast volatility swing (VIX)")
    status.append({"system": "Squeeze Play II-B + fast volatility swing", "data": "VIX", "days": cnt, "needed": int(sl) + 10, "active": ok})

    return sigs, status
