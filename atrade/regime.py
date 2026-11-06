"""Market stress / capitulation detector (Phase 3 research) — information only.

A COINCIDENT detector of market-wide stress and major bottoms, built from a
"correlation to volatility" composite:

    CORR20 = Var(index returns) / mean Var(stock returns)   over 20 trading days
             (rises when stocks fall together — systemic stress)
    VSPIKE = VIX / its own 50-day average   (a volatility spike)
    DD     = drawdown from the trailing 252-day peak
    EXT    = price vs its 200-day average

It flags major market bottoms well (catches the 2018/2020/2022/2025 crashes on
the day) but CANNOT call tops and does NOT predict forward returns — stressed days
are often followed by rebounds. So, like the GEX add-on, it has **no vote in the
rating**; it is shown for context only. Full study and out-of-sample numbers:
`results/sentiment_tuning/PHASE3_FINDINGS.md`.

Everything is causal (uses prices up to the reading date only). Computed from the
shipped `price_data/` panel, so the reading is as of that data's last date.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .config import ROOT

PRICE_DIR = ROOT / "price_data"

# Shared gates (see PHASE3_FINDINGS.md). Bottom presets differ only in the drawdown gate.
VSPIKE_GT = 1.3
CORR20_GT = 0.26
BEAR_VSPIKE_GT = 1.15          # "high-stress regime" label (looser than a bottom)
BEAR_CORR_GT = 0.22
DD_PRESETS = {"major": -0.18, "correction": -0.10}
CORR_WIN = 20


def _panel_closes(dates_from: pd.Series):
    """Close matrix (dates x stocks) aligned to `dates_from.index`, from price_data.
    Index tickers (names starting with '_') are skipped."""
    files = sorted(glob.glob(str(PRICE_DIR / "*.parquet")))
    cols = {}
    for f in files:
        tk = os.path.basename(f)[:-8]
        if tk.startswith("_") or tk == "SPY":
            continue
        try:
            s = pd.read_parquet(f, columns=["Close"])["Close"]
        except Exception:
            continue
        s.index = pd.to_datetime(s.index).normalize()
        cols[tk] = s
    if len(cols) < 50:
        return None
    return pd.DataFrame(cols).reindex(dates_from.index)


def market_stress(recent: int = 120) -> dict | None:
    """Latest market-stress reading plus a short recent history.

    Returns None if the data needed (SPY, VIX, the stock panel) is unavailable.
    """
    try:
        spy = pd.read_parquet(PRICE_DIR / "SPY.parquet", columns=["Close"])["Close"]
        vix = pd.read_parquet(PRICE_DIR / "_VIX.parquet", columns=["Close"])["Close"]
    except Exception:
        return None
    spy.index = pd.to_datetime(spy.index).normalize()
    vix.index = pd.to_datetime(vix.index).normalize()
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    df = pd.DataFrame({"px": spy})
    df["vix"] = vix.reindex(df.index)
    panel = _panel_closes(df["px"])
    if panel is None:
        return None

    r = np.log(df["px"] / df["px"].shift(1))
    stock_ret = np.log(panel / panel.shift(1))
    idx_var = r.rolling(CORR_WIN).var()
    avg_stock_var = stock_ret.rolling(CORR_WIN).var().mean(axis=1)
    df["corr20"] = (idx_var / avg_stock_var).clip(0, 1)
    df["vspike"] = df["vix"] / df["vix"].rolling(50).mean()
    df["dd"] = df["px"] / df["px"].cummax() - 1
    df["ext"] = df["px"] / df["px"].rolling(200).mean() - 1
    df = df.dropna(subset=["vix", "corr20", "vspike", "dd"])
    if df.empty:
        return None

    def bottom(row, preset):
        return bool(row["vspike"] > VSPIKE_GT and row["corr20"] > CORR20_GT and row["dd"] < DD_PRESETS[preset])

    last = df.iloc[-1]
    flags = {p: bottom(last, p) for p in DD_PRESETS}
    stress = bool(last["vspike"] > BEAR_VSPIKE_GT and last["corr20"] > BEAR_CORR_GT)
    hist = df.tail(recent).copy()
    hist["major"] = [bottom(row, "major") for _, row in hist.iterrows()]
    hist["correction"] = [bottom(row, "correction") for _, row in hist.iterrows()]
    return {
        "as_of": df.index[-1],
        "vix": float(last["vix"]), "vspike": float(last["vspike"]),
        "corr20": float(last["corr20"]), "dd": float(last["dd"]), "ext": float(last["ext"]),
        "stress": stress, "flags": flags,
        "n_stocks": int(panel.shape[1]),
        "history": hist[["px", "vix", "vspike", "corr20", "dd", "major", "correction"]],
    }
