"""Parametrized Squeeze Play II-A backtester, vectorized across stocks.

Faithful to atrade/sentiment.py:squeeze_play_2a, but with four tunable knobs:
  entry_X     : oscillator level (fear>+entry_X -> long setup, greed<-entry_X -> short setup)
  exit_X      : opposite-extreme exit level (long exits if osc < -exit_X; short exits if osc > +exit_X)
  trig_long   : price trigger days for longs  (close > prior trig_long-day high)
  trig_short  : price trigger days for shorts (close < prior trig_short-day low)

Stops (max loss, trailing) are kept at the project defaults and NOT tuned.
Entry at next day's open; exit at the close of the exit day. Round-trip cost
applied once per trade (same convention as atrade/validation.py).
"""
from __future__ import annotations
import numpy as np


# ---- single-stock reference implementation (for correctness check) ----------
def backtest_one(osc, open_, high, low, close, entry_X, exit_X, trig_long, trig_short,
                 lb, max_loss, trail_give, trail_delay, cost_rt):
    """Return list of trades (entry_idx, exit_idx, dir, ret_net) on one stock.
    Mirrors squeeze_play_2a's state machine exactly for entries/exits."""
    n = len(close)
    pos = 0; entry = None; peak = 0.0; start = 0; e_idx = None
    trades = []
    tmax = max(lb, trig_long, trig_short, 3)
    for t in range(tmax, n):
        if np.isnan(osc[t]):
            continue
        win = osc[t - lb + 1:t + 1]
        if np.isnan(win).any():
            continue
        # ---- exits (not on entry day or the next day) ----
        if pos != 0 and t > start + 1:
            prof = pos * (close[t] / entry - 1)
            peak = max(peak, prof)
            exit_ = False
            if (pos == 1 and osc[t] < -exit_X) or (pos == -1 and osc[t] > exit_X):
                exit_ = True
            if prof <= -max_loss:
                exit_ = True
            if t - start > trail_delay and peak > 0 and prof <= peak * (1 - trail_give):
                exit_ = True
            if exit_:
                ret = pos * np.log(close[t] / entry) - cost_rt
                trades.append((e_idx, t, pos, ret))
                pos = 0; entry = None
                # fall through so a fresh entry can fire same day
        # ---- entries (only when flat) ----
        if pos == 0:
            hh = np.max(high[t - trig_long:t]) if trig_long > 0 else -np.inf
            ll = np.min(low[t - trig_short:t]) if trig_short > 0 else np.inf
            long_e = win.max() > entry_X and close[t] > hh
            short_e = win.min() < -entry_X and close[t] < ll
            d = 1 if long_e else -1 if short_e else 0
            if d and t + 1 < n and not np.isnan(open_[t + 1]):
                pos = d; entry = open_[t + 1]; e_idx = t + 1; peak = 0.0; start = t
    return trades


# ---- vectorized-across-stocks engine ----------------------------------------
class Panel:
    """Pre-aligned price matrices (days x stocks) + market-wide oscillator."""
    def __init__(self, osc, O, H, L, C, tickers, dates):
        self.osc = osc          # (T,)
        self.O = O; self.H = H; self.L = L; self.C = C   # (T, J)
        self.tickers = tickers; self.dates = dates
        self.T, self.J = C.shape
        self.valid = ~np.isnan(C)            # tradable mask
        self._roll_hi = {}; self._roll_lo = {}

    def rolling_high(self, N):
        """max of High over the N days ending at t-1 (prior N-day high), per (t,j)."""
        if N not in self._roll_hi:
            self._roll_hi[N] = _roll(self.H, N, np.fmax, -np.inf)
            self._roll_lo[N] = _roll(self.L, N, np.fmin, np.inf)
        return self._roll_hi[N], self._roll_lo[N]


def _roll(A, N, op, fill):
    """prior-N-day reduction: out[t] = op(A[t-N:t]); NaNs ignored via fmax/fmin."""
    T, J = A.shape
    out = np.full((T, J), fill)
    # build running window of the N prior rows
    for k in range(1, N + 1):
        shifted = np.full((T, J), fill)
        shifted[k:] = A[:-k]
        out = op(out, np.where(np.isnan(shifted), fill, shifted))
    return out


def sweep_combo(panel: Panel, entry_X, exit_X, trig_long, trig_short, lb,
                max_loss, trail_give, trail_delay, cost_rt, day_lo, day_hi):
    """Run one combo across all stocks for entry days in [day_lo, day_hi).
    Returns per-stock arrays: net log-ret sum, n_long, n_short."""
    osc = panel.osc; O = panel.O; C = panel.C; J = panel.J; T = panel.T
    RH, RL = panel.rolling_high(trig_long)
    RH2, RL2 = panel.rolling_high(trig_short)
    # market-wide setup readiness over the lookback window
    fear = _win_max(osc, lb) > entry_X        # (T,) long setup ready
    greed = _win_min(osc, lb) < -entry_X      # (T,) short setup ready
    oscok = ~np.isnan(osc)
    winok = _win_ok(osc, lb)

    pos = np.zeros(J, np.int8)
    entry = np.full(J, np.nan)
    peak = np.zeros(J)
    start = np.zeros(J, np.int64)
    eidx = np.zeros(J, np.int64)
    ret_sum = np.zeros(J); nlong = np.zeros(J, np.int64); nshort = np.zeros(J, np.int64)

    tmax = max(lb, trig_long, trig_short, 3)
    for t in range(tmax, T - 1):            # need t+1 for entry open
        if not (oscok[t] and winok[t]):
            continue
        ot = osc[t]
        Ct = C[t]
        # exits
        inpos = (pos != 0) & (t > start + 1)
        if inpos.any():
            prof = np.where(inpos, pos * (Ct / entry - 1.0), 0.0)
            peak = np.where(inpos, np.maximum(peak, prof), peak)
            opp = ((pos == 1) & (ot < -exit_X)) | ((pos == -1) & (ot > exit_X))
            mloss = prof <= -max_loss
            trail = (t - start > trail_delay) & (peak > 0) & (prof <= peak * (1 - trail_give))
            do_exit = inpos & (opp | mloss | trail) & ~np.isnan(Ct)
            if do_exit.any():
                r = pos * np.log(Ct / entry) - cost_rt
                # only count trades whose ENTRY fell in the window
                inwin = (eidx >= day_lo) & (eidx < day_hi)
                m = do_exit & inwin
                ret_sum += np.where(m, r, 0.0)
                pos = np.where(do_exit, 0, pos).astype(np.int8)
                entry = np.where(do_exit, np.nan, entry)
        # entries (flat stocks only), within window for entry day t
        if day_lo <= t < day_hi:
            flat = pos == 0
            Ot1 = O[t + 1]
            okopen = ~np.isnan(Ot1) & ~np.isnan(Ct)
            long_e = flat & fear[t] & (Ct > RH[t]) & okopen
            short_e = flat & greed[t] & (Ct < RL2[t]) & okopen & ~long_e
            newd = np.where(long_e, 1, np.where(short_e, -1, 0)).astype(np.int8)
            ent = newd != 0
            if ent.any():
                pos = np.where(ent, newd, pos).astype(np.int8)
                entry = np.where(ent, Ot1, entry)
                eidx = np.where(ent, t + 1, eidx)
                start = np.where(ent, t, start)
                peak = np.where(ent, 0.0, peak)
                nlong += (long_e).astype(np.int64)
                nshort += (short_e).astype(np.int64)
    # close any still-open position at the last valid close inside the window
    openpos = pos != 0
    if openpos.any():
        last = _last_valid(C, day_hi)
        r = pos * np.log(last / entry) - cost_rt
        inwin = (eidx >= day_lo) & (eidx < day_hi)
        m = openpos & inwin & ~np.isnan(r)
        ret_sum += np.where(m, r, 0.0)
    return ret_sum, nlong, nshort


def _win_max(osc, lb):
    T = len(osc); out = np.full(T, -np.inf)
    for k in range(lb):
        s = np.full(T, -np.inf); s[k:] = osc[:T - k] if k else osc
        out = np.fmax(out, s)
    return out

def _win_min(osc, lb):
    T = len(osc); out = np.full(T, np.inf)
    for k in range(lb):
        s = np.full(T, np.inf); s[k:] = osc[:T - k] if k else osc
        out = np.fmin(out, s)
    return out

def _win_ok(osc, lb):
    T = len(osc); bad = np.zeros(T, bool)
    isn = np.isnan(osc)
    for k in range(lb):
        s = np.zeros(T, bool); s[k:] = isn[:T - k] if k else isn
        bad |= s
    return ~bad

def _last_valid(C, day_hi):
    """last non-nan close at or before day_hi-1, per stock."""
    sub = C[:day_hi]
    out = np.full(C.shape[1], np.nan)
    for t in range(sub.shape[0] - 1, -1, -1):
        m = np.isnan(out) & ~np.isnan(sub[t])
        out[m] = sub[t][m]
        if not np.isnan(out).any():
            break
    return out
