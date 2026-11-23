# Phase 3 — correlation-to-volatility model (research)

Exploratory study (not wired into the main `atrade/` engine). Detects market
regimes and turning points from a **correlation-to-volatility composite** — the
realized-correlation proxy (how much stocks move together) combined with the VIX
spike, drawdown and extension. This is the Phase 3 indicator, deliberately
different from Phase 1's volatility-level cash overlay.

Full honest write-up: `results/sentiment_tuning/PHASE3_FINDINGS.md`.
Short version: it is a good **coincident detector of major market bottoms /
stress** (~80–90% of major lows caught out of sample), but **cannot call tops**
and is **not** a reliable forward-return predictor.

## Files

- `phase3.py` — feature builder (`market_features`) and the ZigZag swing filter
  used as ground truth. Run directly to print feature stats and the major turns.
- `phase3_regime.py` — market bull/bear labelling and forward-return evaluation.
- `phase3_turns.py` — tops/bottoms detector, scored (recall/precision/lead-lag)
  against ZigZag turns; writes `phase3_model_turns.json`.
- `phase3_stocks.py` — per-stock capitulation signal and forward-return test.
- `phase3_signal.py` — **run this** to print the current market reading
  (stress regime + bottom flag) from the tuned rules.
- `engine.py`, `driver.py` — shared helpers copied from the sentiment study
  (panel building etc.); only `build_panel_series` is used here.

## Run

```
python research/phase3/phase3_signal.py     # current reading
python research/phase3/phase3.py            # features + ZigZag turns
python research/phase3/phase3_turns.py      # re-score the tops/bottoms detector
```

Needs numpy, pandas, pyarrow, scipy and internet for VIX/price data already in
`price_data/`. Tune/test split is 2016-01-01. All features are causal; the ZigZag
ground truth uses future data for scoring only.
