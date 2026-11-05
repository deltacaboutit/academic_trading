"""The two ways the clues are turned into a forward-looking estimate.

1. **A transparent pressure gauge.** The clues that have a clear meaning are
   averaged into one number, "pressure", running from strongly bottom-like
   (negative) to strongly top-like (positive). It needs no training, so it is
   always available and you can see exactly why it reads the way it does. It is a
   gauge, not a probability.

2. **A trained probability model.** A logistic regression (a standard statistical
   method that weighs several clues and returns a probability between 0 and 1) is
   fitted on history - ideally pooled across many stocks, because true tops and
   bottoms are rare and one stock alone has too few. It outputs, for example, "a
   31% chance this is a top, against a normal rate of 8%". The ratio of those two
   (here about 4x) is the *lift*: how much the estimate beats simply guessing the
   base rate.

The logistic regression is written out by hand with scipy so the tool keeps its
small, fixed set of dependencies (no scikit-learn). It is ridge-regularised (a
penalty that keeps the weights modest) so it behaves with many clues and few
events.

Also here: :func:`expected_range`, which turns the options market's implied
volatility - or, when that is missing, the stock's own recent volatility - into a
forward price range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import indicators as I
from .results import load_json, save_json

# Clues used in the transparent pressure gauge. All are signed so that a positive
# value is "top-like" (stretched up, overbought, tiring at a high, buying
# climax); the same list negated is the bottom-like reading. Clues whose meaning
# is not monotonic (wall distances, which-side-of-flip) are left to the trained
# model only.
PRESSURE_FEATURES = [
    "stretch_ma20", "stretch_ma50", "rsi", "bb_pctb", "ret20_atr",
    "mom_divergence", "trend_decel", "climax", "dist_poc",
]

MODEL_FILE_TOP = "forward_top.json"
MODEL_FILE_BOTTOM = "forward_bottom.json"
MODEL_FILE_CONT = "forward_continuation.json"


# --------------------------------------------------------------- pressure ---

def transparent_pressure(feats: pd.DataFrame) -> pd.Series:
    """Average the clearly-signed clues into one pressure series per day.
    Positive is top-like, negative is bottom-like. Missing clues are simply left
    out of that day's average."""
    cols = [c for c in PRESSURE_FEATURES if c in feats.columns]
    if not cols:
        return pd.Series(np.nan, index=feats.index)
    return feats[cols].mean(axis=1, skipna=True)


# Clues the continuation model reads, over and above the counter-trend pressure.
# These describe the backdrop and are not flipped by trend direction.
CONT_CONTEXT = ["rel_volume", "vix_z", "market_trend", "sector_trend", "rs_market", "corr_sector"]


def continuation_features(feats: pd.DataFrame, state) -> pd.DataFrame:
    """Build the clues for the "will the trend continue?" model.

    The idea is to measure pressure *against* the current trend. In an uptrend a
    reversal looks like a top forming (top-like clues high); in a downtrend a
    reversal looks like a bottom forming (bottom-like clues high). Multiplying
    each signed clue by the trend's direction turns both cases into one "counter-
    trend pressure": a high value always means the trend is under threat,
    whichever way it points. A handful of backdrop clues (volume, market and
    sector trend, fear) are added as they are. The result lines up with a label
    where 1 = the trend reversed.
    """
    state = np.asarray(state, float)
    direction = np.sign(state)
    out = pd.DataFrame(index=feats.index)
    for c in PRESSURE_FEATURES:
        if c in feats.columns:
            out[f"counter_{c}"] = feats[c].to_numpy(float) * direction
    for c in CONT_CONTEXT:
        if c in feats.columns:
            out[c] = feats[c].to_numpy(float)
    return out


def pressure_to_gauge(pressure: float) -> float:
    """Map a pressure value to a friendly 0-100 gauge (50 = neutral). This is a
    readability transform, not a probability."""
    if not np.isfinite(pressure):
        return np.nan
    return float(100.0 / (1.0 + np.exp(-pressure)))


# ------------------------------------------------------ logistic regression --

def _impute(X: np.ndarray) -> np.ndarray:
    """Replace missing clue values with 0. Because the clues are centred (a
    z-score or a signed distance), 0 means "nothing unusual", a safe stand-in
    for a day whose clue could not be computed (for example before the options
    history begins)."""
    X = np.asarray(X, float)
    return np.where(np.isfinite(X), X, 0.0)


def _neg_log_lik(w, X, y, lam):
    z = X @ w
    # stable log(1+exp(z))
    log1p = np.logaddexp(0.0, z)
    ll = np.sum(y * z - log1p)
    reg = lam * np.sum(w[1:] ** 2)          # do not penalise the intercept
    return -ll + reg


def _grad(w, X, y, lam):
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    g = X.T @ (p - y)
    reg = 2 * lam * np.r_[0.0, w[1:]]
    return g + reg


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Fit a ridge-regularised logistic regression. Returns the weight vector
    with the intercept first. ``X`` has one row per day and one column per clue
    (missing values imputed); ``y`` is 0/1."""
    X = _impute(X)
    y = np.asarray(y, float)
    Xb = np.c_[np.ones(len(X)), X]
    w0 = np.zeros(Xb.shape[1])
    res = minimize(_neg_log_lik, w0, args=(Xb, y, l2), jac=_grad, method="L-BFGS-B")
    return res.x


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Probabilities from a fitted weight vector."""
    Xb = np.c_[np.ones(len(X)), _impute(X)]
    return 1.0 / (1.0 + np.exp(-(Xb @ w)))


# ----------------------------------------------------- train / save / load ---

def train_binary(X: np.ndarray, y: np.ndarray, feature_names: list[str], l2: float = 1.0) -> dict:
    """Fit one yes/no model (top, or bottom) and return a saveable dict with the
    weights, the clue names, and the base rate it was trained on."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok].astype(float)
    w = fit_logit(X, y, l2)
    return {"kind": "logit", "features": list(feature_names), "intercept": float(w[0]),
            "coef": [float(v) for v in w[1:]], "base_rate": float(y.mean()), "n": int(len(y)),
            "n_pos": int(y.sum()), "l2": float(l2)}


def apply_model(model: dict, feats: pd.DataFrame) -> np.ndarray:
    """Run a saved model over a feature table, lining the clues up by name and
    filling any the table does not have with 0 (neutral)."""
    cols = model["features"]
    X = np.column_stack([feats[c].to_numpy(float) if c in feats.columns
                         else np.full(len(feats), np.nan) for c in cols])
    w = np.r_[model["intercept"], model["coef"]]
    return predict_proba(w, X)


def save_model(model: dict, name: str):
    return save_json(model, name)


def load_model(name: str) -> dict | None:
    return load_json(name)


def lift(prob: float, base_rate: float) -> float:
    """How many times more likely than the base rate. 1x = no better than
    guessing the long-run average."""
    if not np.isfinite(prob) or not base_rate or base_rate <= 0:
        return np.nan
    return float(prob / base_rate)


# --------------------------------------------------------- expected range ----

def expected_range(df: pd.DataFrame, sentiment: dict | None = None, horizon: int = 20,
                   n_sigma: float = 1.0, iv_annual: float | None = None) -> dict:
    """A forward price range over ``horizon`` trading days, centred on today's
    close and ``n_sigma`` standard deviations wide.

    The width is the move the options market is pricing in when that is
    available, and otherwise the stock's own recent (realised) volatility.
    ``iv_annual`` is the preferred input: the 30-day at-the-money implied
    volatility index (``ivx_30d``), which is the clean "how big a move is the
    market pricing" number. We avoid the plain average implied volatility, which
    is pulled up by far out-of-the-money options (the "volatility smile") and
    would make the range too wide."""
    close = float(df["Close"].iloc[-1])
    iv = np.nan
    source = "realised volatility"
    if iv_annual is not None and np.isfinite(iv_annual) and iv_annual > 0:
        iv = float(iv_annual)
        source = "options-implied volatility (30-day ATM)"
    if not np.isfinite(iv):
        ret = df["Close"].pct_change().dropna()
        iv = float(ret.tail(63).std() * np.sqrt(252)) if len(ret) >= 20 else np.nan
    if not np.isfinite(iv) or iv <= 0:
        return {"low": np.nan, "high": np.nan, "sigma_pct": np.nan, "source": source}
    sigma = iv * np.sqrt(horizon / 252.0)       # scale annual vol to the horizon
    return {"low": close * (1 - n_sigma * sigma), "high": close * (1 + n_sigma * sigma),
            "sigma_pct": float(sigma), "source": source, "close": close, "horizon": horizon}
