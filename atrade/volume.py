"""Framework 5 - volume confirmation."""
from __future__ import annotations

import numpy as np


def relative_volume(vol: np.ndarray, days: int = 50) -> np.ndarray:
    """RV_t = V_t / average(V_{t-1}, ..., V_{t-days})."""
    v = np.asarray(vol, float)
    out = np.full(len(v), np.nan)
    c = np.cumsum(np.insert(np.nan_to_num(v), 0, 0.0))
    for t in range(days, len(v)):
        avg = (c[t] - c[t - days]) / days
        if avg > 0:
            out[t] = v[t] / avg
    return out


def volume_weight(rv: float, s: dict) -> float:
    """1 (confirmed), 0.5 (neutral) or 0 (unconfirmed)."""
    if rv is None or np.isnan(rv):
        return 0.5          # no volume data: neutral (tool choice)
    if rv >= s["rv_full"]:
        return 1.0
    if rv >= s["rv_half"]:
        return 0.5
    return 0.0
