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
- **Other volatility indices** (`results/sentiment_tuning/alt_indices_scan.csv`,
  summarised in FINDINGS.md): the same sweep on VXN, VXO, VXV, VXD, RVX, VVIX, OVX,
  GVZ, the VXN−VIX spread and the VXV/VIX term structure — none beats buy-and-hold.
- **Volatility regime overlay** (`results/sentiment_tuning/REGIME_OVERLAY.md`): a
  VIX-trend + term-structure signal that moves to cash in high-risk regimes. It
  reliably cuts drawdowns (~−50%→−26% in 2008; ~80% of stocks improve) but roughly
  halves return, so it is a risk dial, not a return booster. Research only, not
  wired into the rating engine.
- **Sector volatility sources** (`results/sentiment_tuning/SECTOR_VOL_SOURCES.md`):
  inventory of per-sector volatility trackers. Dedicated CBOE implied-vol indices
  exist only for a few (VXN→tech/comm, OVX→energy, GVZ→materials, VXD→industrials,
  RVX→small caps); all other sectors must use realized vol from the SPDR ETF
  (XLF/XLV/XLY/XLP/XLU/XLRE…).
- **Sector-matched overlay** (`results/sentiment_tuning/SECTOR_OVERLAY.md`): timing
  each stock by its own sector's volatility (VXN for tech/comm, realized ETF vol
  elsewhere) modestly beats the market-wide VIX overlay on drawdown (market −51%
  B&H → −35% tune / −13% test at W=100) while keeping slightly more return, but
  still roughly halves return vs buy-and-hold. A risk dial, research only.
