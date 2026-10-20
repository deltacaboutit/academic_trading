"""Stored research results (folder results/ in the repo).

    results/bootstrap/<TICKER>.json     Framework 9 step 3 for one stock (null-model fits + p-values)
    results/random_patterns/<TICKER>.csv Framework 9 step 4 for one stock
    results/lmw_settings.json           LMW bandwidth (calibrated on training data) and the
                                        pattern directions learned from the 9b tests
    results/tuned_weights.json          rating-engine weights tuned on training data
    results/summary/*.csv               cross-stock tables (Bernoulli tests, 9b by exchange, ...)

Written by the app ("Run full validation") or by the research runner / GitHub Action;
read by the app. Everything is plain JSON / CSV so it is easy to inspect on GitHub.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

RESULTS = ROOT / "results"


def _fname(ticker: str) -> str:
    return ticker.upper().replace("^", "_").replace("/", "-")


def save_bootstrap(ticker: str, boot: dict, root: Path = RESULTS) -> Path:
    p = root / "bootstrap" / f"{_fname(ticker)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    fits = boot["fits"].replace({np.nan: None}).to_dict("records")
    payload = {"ticker": ticker.upper(), "B": boot["B"], "testable": boot["testable"], "rules": boot.get("rules", []),
               "fits": fits, "results": boot["results"].replace({np.nan: None}).to_dict("records"),
               "created": pd.Timestamp.now().isoformat(timespec="minutes")}
    p.write_text(json.dumps(payload, default=str))
    return p


def load_bootstrap(ticker: str, root: Path = RESULTS) -> dict | None:
    p = root / "bootstrap" / f"{_fname(ticker)}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["fits"] = pd.DataFrame(d["fits"]); d["results"] = pd.DataFrame(d["results"])
    return d


def save_table(df: pd.DataFrame, name: str, root: Path = RESULTS) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_table(name: str, root: Path = RESULTS) -> pd.DataFrame | None:
    p = root / name
    return pd.read_csv(p) if p.exists() else None


def load_json(name: str, root: Path = RESULTS) -> dict | None:
    p = root / name
    return json.loads(p.read_text()) if p.exists() else None


def save_json(obj: dict, name: str, root: Path = RESULTS) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def lmw_settings() -> dict:
    """Bandwidth and learned directions; falls back to the defaults."""
    from .lmw import CLASSIC_DIRECTION, DEFAULT_H
    d = load_json("lmw_settings.json") or {}
    return {"h": float(d.get("h", DEFAULT_H)), "directions": d.get("directions", CLASSIC_DIRECTION),
            "source": "research results" if d else "defaults (classic directions for HS/IHS/DTOP/DBOT)"}
