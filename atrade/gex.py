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
import pandas as pd

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


# ------------------------------------------------ daily levels (mixed method) ---
# Stored every day in the options summary. Two kinds of level, each computed the way it
# is normally defined (decided 2026-10-21, see README "GEX levels"):
#
#   WALLS - at STRIKES (the industry definition; SpotGamma: "the strike with the largest net
#   call/put gamma"). For every contract at today's spot S:
#       GEX = sign * phi(d1)/(S iv sqrt T) * OI * 100 * S^2 * 0.01      (dollar gamma per 1% move)
#   summed per strike, calls and puts separately.
#       Call wall = strike with the largest total call GEX.
#       Put wall  = strike with the most negative total put GEX.
#   Contracts: expiries <= 90 days, open interest >= 5. d1 includes the 3-month T-bill rate r.
#
#   FLIP and TOTAL GEX - from the PRICE PROFILE (Watchtower's method). The flip is where total
#   GEX changes sign as the price moves, so it can only be found by re-pricing the whole chain
#   at many hypothetical prices:
#   1. Coarse grid 0.5-1.5 x spot (100 points, r = 4.5%): rough levels, used only to size step 2.
#   2. Fine grid of 200 points from min(0.85 S, 0.95 coarse put level, 0.95 flip) to
#      max(1.15 S, 1.05 coarse call level, 1.05 flip), at the day's rate r.
#      Flip = last STABLE sign change (>= 5 grid points on each side; otherwise the last sign change).
#      Total GEX = the profile's value at spot.
#
# Watchtower placed the walls on the profile too (highest point above the flip, lowest below it),
# which gives prices between strikes (e.g. 334.95). Its flip and total GEX are kept as they were.

def _profile(prices, K, T, iv, oi, is_call, r):
    S = np.asarray(prices, float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        g = np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * iv * np.sqrt(T))
    return np.sum(np.nan_to_num(g) * oi * 100 * S ** 2 * 0.01 * np.where(is_call, 1, -1), axis=1)


def _flip_simple(prices, gex):
    sc = np.where(np.diff(np.sign(gex)))[0]
    if not len(sc):
        return None
    i = int(sc.max())
    return prices[i] - gex[i] * (prices[i + 1] - prices[i]) / (gex[i + 1] - gex[i])


def find_flip(prices, gex, w=5):
    s = np.sign(gex).astype(int)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], len(s)]
    lens = ends - starts; signs = s[starts]
    ok = (signs[1:] * signs[:-1] == -1) & (lens[:-1] >= w) & (lens[1:] >= w)
    if not ok.any():
        return _flip_simple(prices, gex)
    i = starts[np.where(ok)[0][-1] + 1] - 1
    return prices[i] - gex[i] * (prices[i + 1] - prices[i]) / (gex[i + 1] - gex[i])


def strike_walls(c, spot, r=0.045):
    """Call and put walls at strikes (see the notes above). c: contracts already filtered."""
    out = {"call_wall_price": np.nan, "call_wall_gex": np.nan, "put_wall_price": np.nan, "put_wall_gex": np.nan}
    if c.empty:
        return out
    K, T, iv = c["strike"].to_numpy(float), c["T"].to_numpy(float), c["iv"].to_numpy(float)
    isc = c["is_call"].to_numpy(bool)
    S = float(spot)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        g = np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * iv * np.sqrt(T))
    gex = np.nan_to_num(g) * c["oi"].to_numpy(float) * 100 * S * S * 0.01 * np.where(isc, 1, -1)
    by = pd.DataFrame({"strike": K, "call": isc, "gex": gex}).groupby(["call", "strike"])["gex"].sum()
    if True in by.index.get_level_values(0):
        cs = by.loc[True]
        out["call_wall_price"], out["call_wall_gex"] = float(cs.idxmax()), float(cs.max())
    if False in by.index.get_level_values(0):
        ps = by.loc[False]
        out["put_wall_price"], out["put_wall_gex"] = float(ps.idxmin()), float(ps.min())
    return out


def _filtered(chain_one):
    return chain_one[(chain_one["oi"] >= 5) & chain_one["iv"].notna() & (chain_one["iv"] > 0) & (chain_one["T"] > 0)]


def daily_levels(chain_one, spot, r=0.045):
    """GEX levels for one stock on one day: walls at strikes, flip and total from the price profile."""
    out = {"gex_total": np.nan, "gex_flip": np.nan, "call_wall_price": np.nan, "call_wall_gex": np.nan,
           "put_wall_price": np.nan, "put_wall_gex": np.nan, "wall_method": "strike"}
    c = _filtered(chain_one)
    if c.empty or not np.isfinite(spot) or spot <= 0:
        return out
    K, T, iv = c["strike"].to_numpy(float), c["T"].to_numpy(float), c["iv"].to_numpy(float)
    oi, isc = c["oi"].to_numpy(float), c["is_call"].to_numpy(bool)
    coarse = np.linspace(spot * 0.5, spot * 1.5, 100)
    gc = _profile(coarse, K, T, iv, oi, isc, 0.045)
    cwp, pwp = int(coarse[np.argmax(gc)]), int(coarse[np.argmin(gc)])
    fl = find_flip(coarse, gc)
    fl_ = fl if fl is not None else spot
    prices = np.linspace(min(spot * 0.85, pwp * 0.95, fl_ * 0.95), max(spot * 1.15, cwp * 1.05, fl_ * 1.05), 200)
    g = _profile(prices, K, T, iv, oi, isc, r)
    flip = find_flip(prices, g)
    out["gex_total"] = float(np.interp(spot, prices, g))
    out["gex_flip"] = float(flip) if flip is not None else np.nan
    out.update(strike_walls(c, spot, r))
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
