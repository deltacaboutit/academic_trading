"""[GEX] Gamma exposure - an ADD-ON that is not in either book or the framework.

Like the LMW items, GEX is kept separate: it is drawn on the chart and
back-tested, but it has NO vote in the official rating unless the user
decides otherwise after testing.

Gamma per contract (Black-Scholes, no rates or dividends):
    gamma = phi(d1) / (S * iv * sqrt(T)),   d1 = (ln(S/K) + iv^2 T / 2) / (iv sqrt(T))
GEX per contract = sign * gamma * OI * 100 * S^2 * GEX_SCALE, sign = +1 call, -1 put
(dealers assumed long calls and short puts - the usual convention).
  Call wall = strike with the largest total call GEX.
  Put wall  = strike with the most negative total put GEX.
  Flip      = spot level where total GEX changes sign (nearest to today's spot),
              found by re-pricing gamma on a grid of hypothetical spot prices.
GEX_SCALE is calibrated against the user's own gex_history.csv on import.
"""
from __future__ import annotations

import math

import numpy as np

from .signals import Signal

GEX_SCALE = 1.0          # replaced by calibration (see options_store.calibrate_gex)
IV_MIN, IV_MAX = 0.01, 5.0


def gamma(S, K, T, iv):
    S = np.asarray(S, float); K = np.asarray(K, float); T = np.asarray(T, float); iv = np.asarray(iv, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = iv * np.sqrt(T)
        d1 = (np.log(S / K) + 0.5 * iv * iv * T) / sq
        g = np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sq)
    return np.where(np.isfinite(g), g, 0.0)


def usable(df):
    """Contracts that can be priced: positive time, sensible IV, some open interest."""
    return df[(df["T"] > 0) & (df["iv"] > IV_MIN) & (df["iv"] < IV_MAX) & (df["oi"] > 0)]


def levels(chain_one, spot, scale=None):
    """GEX levels for one stock on one day. chain_one needs strike, is_call, iv, oi, T."""
    scale = GEX_SCALE if scale is None else scale
    out = {"gex_total": np.nan, "gex_flip": np.nan, "call_wall_price": np.nan, "call_wall_gex": np.nan,
           "put_wall_price": np.nan, "put_wall_gex": np.nan}
    c = usable(chain_one)
    if c.empty or not np.isfinite(spot) or spot <= 0:
        return out
    sign = np.where(c["is_call"].to_numpy(bool), 1.0, -1.0)
    K = c["strike"].to_numpy(float); T = c["T"].to_numpy(float); iv = c["iv"].to_numpy(float)
    oi = c["oi"].to_numpy(float)
    g = sign * gamma(spot, K, T, iv) * oi * 100 * spot * spot * scale
    out["gex_total"] = float(g.sum())
    calls = c.assign(g=g)[sign > 0].groupby("strike")["g"].sum()
    puts = c.assign(g=g)[sign < 0].groupby("strike")["g"].sum()
    if len(calls):
        out["call_wall_price"] = float(calls.idxmax()); out["call_wall_gex"] = float(calls.max())
    if len(puts):
        out["put_wall_price"] = float(puts.idxmin()); out["put_wall_gex"] = float(puts.min())
    grid = np.linspace(0.7 * spot, 1.3 * spot, 241)
    tot = (gamma(grid[:, None], K[None, :], T[None, :], iv[None, :]) * (sign * oi)[None, :]).sum(axis=1) * grid ** 2
    sc = np.where(np.sign(tot[:-1]) * np.sign(tot[1:]) < 0)[0]
    if len(sc):
        xs = [grid[i] - tot[i] * (grid[i + 1] - grid[i]) / (tot[i + 1] - tot[i]) for i in sc]
        out["gex_flip"] = float(min(xs, key=lambda v: abs(v - spot)))
    return out


# ------------------------------------------------------------- signals -----

def gex_signals(df, lv: dict) -> list[Signal]:
    """Back-test-only signals. Levels from the PREVIOUS day are used on day t, so
    nothing is used before it was known (the levels need the day's close)."""
    c = df["Close"].to_numpy(float); h = df["High"].to_numpy(float); l = df["Low"].to_numpy(float)
    out = []

    def prev(a):
        p = np.full(len(a), np.nan); p[1:] = a[:-1]; return p

    pw, cw, fl = prev(lv["put_wall_price"]), prev(lv["call_wall_price"]), prev(lv["gex_flip"])
    fl2 = prev(fl)
    with np.errstate(invalid="ignore"):
        buy_pw = (l <= pw) & (c > pw)
        sell_cw = (h >= cw) & (c < cw)
        up = (c > fl) & (prev(c) < fl2)
        dn = (c < fl) & (prev(c) > fl2)
    def add(mask, rule, d, text):
        for t in np.where(mask)[0]:
            out.append(Signal(rule, "gex", "addon", d, int(t), weight=0.0, vol_adjust=False, text=text))
    add(buy_pw, "[GEX] Put-wall hold", +1, "Price dipped to the put wall and closed above it.")
    add(sell_cw, "[GEX] Call-wall rejection", -1, "Price reached the call wall and closed below it.")
    add(up, "[GEX] Flip cross", +1, "Close crossed above the gamma flip level.")
    add(dn, "[GEX] Flip cross", -1, "Close crossed below the gamma flip level.")
    return out
