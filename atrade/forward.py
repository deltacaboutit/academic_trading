"""The forward-looking estimate: type a ticker, get the three indicators plus a
correlation read-out and a forward price range.

This ties the pieces together. For the latest day it reports:

* **Top risk** and **bottom chance** - the trained models' probabilities, each
  next to the normal (base) rate and the resulting "lift" (how many times more
  likely than usual).
* **Trend continuation** - the odds the current up- or down-trend continues,
  reverses, or stalls over the next few weeks. (This indicator is weak - see the
  note it carries - and is shown for context.)
* **A pressure gauge** - the transparent, always-available reading of how
  stretched the stock is, from bottom-like to top-like.
* **Expected range** - where the options market (or, failing that, recent
  volatility) suggests the price could sit over the horizon.
* **Key levels** - put wall, gamma flip, call wall and the volume point of
  control.
* **Why** - the clues pushing the reading most, in plain words.
* **Correlation analysis** - the stock's sector, how it tracks its sector and
  the market, and which sectors are strong or weak.

Everything degrades gracefully: with no trained model it still shows the gauge
and range; with no options or sector data those parts are simply omitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import forward_features as FF
from . import forward_model as M
from . import sectors as S
from .cleaning import clean
from .config import DEFAULTS
from .trend import trend_state

# Plain-language names for the clues, used when explaining a reading.
FEATURE_LABELS = {
    "stretch_ma20": "distance above/below the 20-day average",
    "stretch_ma50": "distance above/below the 50-day average",
    "rsi": "RSI (overbought/oversold)",
    "bb_pctb": "position in the Bollinger band",
    "ret20_atr": "size of the last 20-day move",
    "mom_divergence": "momentum diverging from price",
    "trend_decel": "trend losing speed",
    "climax": "heavy volume at the extreme",
    "dist_poc": "distance from the heavily-traded price",
    "at_high60": "sitting at a 60-day high",
    "at_low60": "sitting at a 60-day low",
    "rel_volume": "unusually heavy volume",
    "obv_slope": "on-balance-volume trend",
    "to_call_wall": "nearness to the call wall (ceiling)",
    "to_put_wall": "nearness to the put wall (floor)",
    "below_flip": "below the gamma flip (moves amplified)",
    "dist_flip": "distance from the gamma flip",
    "pc_volume_z": "put/call volume (fear)",
    "pc_oi_z": "put/call open interest (fear)",
    "iv_z": "implied volatility level (fear)",
    "iv_change": "implied volatility jump",
    "vix_z": "market fear gauge (VIX)",
    "rs_sector": "strength vs its sector",
    "corr_sector": "how tightly it tracks its sector",
    "sector_trend": "its sector's own trend",
    "rs_market": "strength vs the market",
    "corr_market": "how tightly it tracks the market",
    "market_trend": "the market's own trend",
}
for c in list(FEATURE_LABELS):
    FEATURE_LABELS[f"counter_{c}"] = "counter-trend " + FEATURE_LABELS[c]


def _latest_ivx(ticker: str) -> float | None:
    """The most recent 30-day at-the-money implied-volatility index for the
    stock, read straight from the options summary. Returns ``None`` when there is
    no options history."""
    try:
        from . import options_store as OS
        df = OS.read_summary([ticker.upper()])
        if df is None or df.empty or "ivx_30d" not in df:
            return None
        v = df[df["ticker"] == ticker.upper()].sort_values("date")["ivx_30d"].dropna()
        if not len(v):
            return None
        iv = float(v.iloc[-1])
        return iv / 100.0 if iv > 3 else iv      # stored in percent (e.g. 30 = 30%)
    except Exception:
        return None


def _load_inputs(ticker: str, s: dict, refresh: bool, prices=None):
    if prices is None:
        raw, source = D.load_prices(ticker, int(s.get("history_years", 20)), refresh)
    else:
        raw, source = prices, "supplied data"
    df, _, _ = clean(raw, s)
    state = trend_state(df["Close"].to_numpy(float), s)
    try:
        gex = D.load_gex(ticker, df.index)
    except Exception:
        gex = None
    try:
        sent = D.load_sentiment(ticker, df.index, int(s.get("history_years", 20)))
    except Exception:
        sent = {}
    sector_closes = S.load_sector_closes()
    asg = S.assign_sector(ticker, df["Close"], sector_closes=sector_closes)
    etf = sector_closes.get(asg["etf"]) if asg["etf"] else None
    spy = S._close(S.MARKET_ETF)
    feats = FF.build_features(df, ticker, gex=gex, sentiment=sent,
                              sector_etf_close=etf, market_close=spy)
    return df, state, gex, sent, feats, asg, sector_closes, spy


def _reasons(model: dict | None, feats_row: pd.Series, k: int = 4, want_sign: int = +1):
    """The clues pushing a reading most, newest value times its weight. ``want_sign``
    +1 lists what pushes the probability UP, -1 what pushes it down."""
    if model is None:
        return []
    out = []
    for f, c in zip(model["features"], model["coef"]):
        v = feats_row.get(f, np.nan)
        if not np.isfinite(v):
            continue
        contrib = c * v
        out.append((f, contrib, v))
    out = [o for o in out if np.sign(o[1]) == want_sign and abs(o[1]) > 1e-6]
    out.sort(key=lambda x: -abs(x[1]))
    res = []
    for f, contrib, v in out[:k]:
        label = FEATURE_LABELS.get(f, f)
        res.append({"feature": f, "label": label, "contribution": float(contrib), "value": float(v)})
    return res


def _key_levels(gex: dict | None, feats: pd.DataFrame, df: pd.DataFrame) -> dict:
    out = {}
    if gex is not None:
        for key, nice in (("put_wall_price", "put_wall"), ("gex_flip", "gamma_flip"),
                          ("call_wall_price", "call_wall")):
            arr = gex.get(key)
            if arr is not None:
                v = pd.Series(arr).dropna()
                if len(v):
                    out[nice] = float(v.iloc[-1])
    # point of control from the volume-profile feature path
    from .forward_features import _rolling_poc
    from . import indicators as I
    atr = I.atr_filled(df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float), 14)
    poc = _rolling_poc(df["High"].to_numpy(float), df["Low"].to_numpy(float),
                       df["Volume"].to_numpy(float), atr, 60, 50)
    pp = pd.Series(poc).dropna()
    if len(pp):
        out["point_of_control"] = float(pp.iloc[-1])
    return out


def forward_estimate(ticker: str, s: dict | None = None, refresh: bool = False,
                     prices=None, horizon: int = 20) -> dict:
    """The full forward-looking card for one stock (latest day)."""
    s = dict(s or DEFAULTS)
    ticker = ticker.upper().strip()
    df, state, gex, sent, feats, asg, sector_closes, spy = _load_inputs(ticker, s, refresh, prices)
    cont_feats = M.continuation_features(feats, state)
    last = feats.iloc[-1]
    cur_state = int(state[-1])

    top_m = M.load_model(M.MODEL_FILE_TOP)
    bot_m = M.load_model(M.MODEL_FILE_BOTTOM)
    con_m = M.load_model(M.MODEL_FILE_CONT)

    def prob_last(model, table):
        if model is None:
            return np.nan
        return float(M.apply_model(model, table.iloc[[-1]])[0])

    top_p = prob_last(top_m, feats)
    bot_p = prob_last(bot_m, feats)

    # continuation -> continue / reverse / stall
    cont = {"state": cur_state}
    if con_m is not None and cur_state != 0:
        rev_given = float(M.apply_model(con_m, cont_feats.iloc[[-1]])[0])
        stall = float(con_m.get("stall_rate", 0.08))
        cont.update({"continue": (1 - stall) * (1 - rev_given), "reverse": (1 - stall) * rev_given,
                     "stall": stall, "reverse_given_resolved": rev_given})

    pressure = float(M.transparent_pressure(feats).iloc[-1])
    rng = M.expected_range(df, sent, horizon=horizon, iv_annual=_latest_ivx(ticker))
    levels = _key_levels(gex, feats, df)

    out = {
        "ticker": ticker, "asof": df.index[-1].date(), "close": float(df["Close"].iloc[-1]),
        "trend_state": cur_state,
        "top": {"prob": top_p, "base_rate": top_m["base_rate"] if top_m else np.nan,
                "lift": M.lift(top_p, top_m["base_rate"]) if top_m else np.nan,
                "reasons": _reasons(top_m, last, want_sign=+1)},
        "bottom": {"prob": bot_p, "base_rate": bot_m["base_rate"] if bot_m else np.nan,
                   "lift": M.lift(bot_p, bot_m["base_rate"]) if bot_m else np.nan,
                   "reasons": _reasons(bot_m, last, want_sign=+1)},
        "continuation": cont,
        "continuation_reasons": _reasons(con_m, cont_feats.iloc[-1], want_sign=+1) if cur_state != 0 else [],
        "pressure": pressure, "gauge": M.pressure_to_gauge(pressure),
        "expected_range": rng, "key_levels": levels,
        "correlation": S.correlation_report(ticker, df["Close"]),
        "models_available": {"top": top_m is not None, "bottom": bot_m is not None,
                             "continuation": con_m is not None},
        "df": df, "features": feats,
    }
    return out


# ------------------------------------------------------------ plain text ----

def render_text(est: dict) -> str:
    """A compact plain-English summary of the estimate (for the terminal or a
    quick read)."""
    L = []
    t = est
    L.append(f"{t['ticker']}  (as of {t['asof']}, close {t['close']:.2f})")
    state_name = {1: "uptrend", -1: "downtrend", 0: "sideways"}[t["trend_state"]]
    L.append(f"Trend now: {state_name}")

    def pct(x):
        return f"{x:.0%}" if np.isfinite(x) else "n/a"

    tp = t["top"]
    if np.isfinite(tp["prob"]):
        L.append(f"Top risk (next ~20d): {pct(tp['prob'])}  (normal {pct(tp['base_rate'])}, "
                 f"{tp['lift']:.1f}x)" if np.isfinite(tp['lift']) else f"Top risk: {pct(tp['prob'])}")
        if tp["reasons"]:
            L.append("   because: " + "; ".join(r["label"] for r in tp["reasons"]))
    bp = t["bottom"]
    if np.isfinite(bp["prob"]):
        L.append(f"Bottom chance: {pct(bp['prob'])}  (normal {pct(bp['base_rate'])}, "
                 f"{bp['lift']:.1f}x)" if np.isfinite(bp['lift']) else f"Bottom chance: {pct(bp['prob'])}")
        if bp["reasons"]:
            L.append("   because: " + "; ".join(r["label"] for r in bp["reasons"]))
    c = t["continuation"]
    if "continue" in c:
        L.append(f"Trend: {pct(c['continue'])} continues / {pct(c['reverse'])} reverses / "
                 f"{pct(c['stall'])} stalls  (this indicator is weak - near a coin flip)")
    r = t["expected_range"]
    if np.isfinite(r.get("low", np.nan)):
        L.append(f"Expected ~20-day range: {r['low']:.2f} to {r['high']:.2f}  (from {r['source']})")
    if t["key_levels"]:
        L.append("Key levels: " + ", ".join(f"{k.replace('_', ' ')} {v:.2f}"
                                             for k, v in t["key_levels"].items()))
    L.append(f"Pressure gauge: {t['gauge']:.0f}/100  (50 = neutral, high = top-like)")
    cr = t["correlation"]["sector"]
    if cr.get("etf"):
        L.append(f"Sector (by correlation): {cr['name']} [{cr['etf']}], corr {cr['corr']:.2f}")
    return "\n".join(L)
