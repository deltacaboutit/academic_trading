# CLAUDE.md — working notes for this repo

Academic Trading is a personal stock-analysis tool built from two technical-analysis
books plus PatternFinder. The engine lives in `atrade/`, one file per "Framework";
`app.py` is the Streamlit browser app. See `README.md` for the full tour.

## Layout

- `atrade/` — engine. cleaning.py (F1), turning_points.py (F2), trend.py (F3),
  zones/patterns/gauges.py (F4), volume.py (F5), **sentiment.py (F6)**,
  rating.py (F7), trades.py (F8), validation.py (F9); gex.py and lmw.py are add-ons.
- `atrade/config.py` — **all tunable settings** live in the `DEFAULTS` dict.
- `price_data/` — ~25 years of daily prices per stock (parquet). `_VIX.parquet` is the VIX.
- `options_data/summary/` — per-stock options sentiment, but only ~5 months so far.
- `results/` — research outputs. `tests/` — formula, look-ahead and data checks.

## Conventions (keep these)

- **No look-ahead.** A value on day *t* uses only prices up to *t*; entries are the
  next day's open. There are tests that enforce this — do not break them.
- **Costs always included.** Round-trip cost = `2*(cost_one_way + spread_one_way)`.
- **Validate by time split**, comparing to buy-and-hold (`atrade/validation.py`).
  Tune on earlier data, test on later data; never tune on the test part.
- The tool is honest by design: both books found most technical rules do not beat
  buy-and-hold after costs. Report results that way — do not oversell.
- Run the checks with `python -m pytest tests/` (needs numpy, pandas, scipy, pyarrow).

## Research findings

- **Framework 6 (sentiment) tuning study → `results/sentiment_tuning/FINDINGS.md`.**
  Tuned the Squeeze Play II-A entry threshold, exit threshold and long/short
  price-trigger days (`sp2a_entry`, `sp2a_exit`, `sp2a_trig_long`, `sp2a_trig_short`
  in config.py) on long-history market-wide sentiment (VIX and the Cboe equity
  put/call ratio). **No setting beat buy-and-hold in or out of sample.** The
  extended system this produced is off by default (`sp2a_extended_enable`) and,
  when on, is back-tested only (does not vote). Full tables are in that folder.
