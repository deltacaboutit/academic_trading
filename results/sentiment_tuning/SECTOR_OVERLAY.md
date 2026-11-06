# Sector-matched regime overlay vs market-wide VIX overlay

*Run 2026-11-03. Code: `scratchpad/sector_overlay.py`; results: `sector_overlay.csv`.*

## Idea

Phase 1 timed every stock with one market-wide signal (VIX). Here each stock is
timed by **its own sector's** volatility instead: risk-off (go to cash) when the
sector tracker rises above its own W-day moving average. Technology and
Communication Services use **VXN** (implied); every other sector uses the
**realized volatility of its SPDR sector ETF** (XLF, XLV, XLY, XLP, XLI, XLB,
XLE, XLU, XLRE). 486 of 516 stocks mapped to a GICS sector (94%). Long only,
0.1% per switch, tune < 2018 / test ≥ 2018.

## Result — sector-matching modestly beats market-wide timing (for risk)

Equal-weight market portfolio of the mapped stocks:

| Window | Metric | Buy & hold | VIX overlay | Sector overlay |
|---|---|---|---|---|
| W=20 | drawdown tune / test | −51% / −38% | −39% / **−14%** | −34% / −20% |
| W=50 | drawdown tune / test | −51% / −38% | −41% / −23% | **−34% / −15%** |
| W=100 | drawdown tune / test | −51% / −38% | −46% / −22% | **−35% / −13%** |
| W=100 | return tune / test | 18% / 17% | 8% / 8% | 9% / 9% |
| W=100 | Calmar tune / test | 0.35 / 0.44 | 0.17 / 0.36 | **0.26 / 0.70** |

Across individual stocks, the sector overlay beat the VIX overlay on worst
drawdown for ~57–67% of stocks (W=50/100) and on return for ~55–62% (tune; mixed
in test). The improvement is clearest at the **longer windows (W=100)**: lower
drawdown than VIX in both eras, with slightly higher return.

## Honest reading

- **Sector-matching is a real, if modest, improvement** over the one-size VIX
  overlay for the risk-reduction goal: better drawdown control and slightly
  better return retention, most consistently at W=100.
- **The fundamental trade-off is unchanged.** Both overlays still roughly halve
  the return versus buy-and-hold (≈9% vs ≈17–18%). Sector-matching does not beat
  buy-and-hold on return, and only sometimes on Calmar (tune still favours B&H).
- **Implied reacts faster than realized.** At the short window during the 2020
  crash, the market-wide *implied* VIX (−14%) protected better than sector
  *realized* vol (−20%) — realized vol lags a sudden shock. At longer windows,
  the sector signal wins. A hybrid (implied where it exists, realized elsewhere;
  short window for shocks, long for regimes) is the natural refinement.

## Recommended configuration

For drawdown control, **sector-matched with W=100** is the best, most stable
setting: market drawdown −35% (tune) / −13% (test) vs buy-and-hold −51% / −38%,
while keeping ~9% annual return. Still a **risk dial, not a return booster.**

## Caveats

- Mixes implied (VXN) and realized (ETF) trackers; the regime rule is relative to
  each series' own trend, so this is acceptable but not perfectly uniform.
- Realized sector vol is derived from stock prices, so it is partly circular as a
  "sentiment" input (fine as a volatility-regime input).
- Real Estate (XLRE) tracker only starts 2015; those stocks are mostly un-timed
  before then. 6% of stocks had no GICS match and were excluded.
- Cash earns 0% (no interest), slightly understating the overlays.

## Hybrid refinement (2026-11-03) — `sector_hybrid.csv`

Two refinements were tested (equal-weight market portfolio, same split):

| Config | CAGR tune/test | Max drawdown tune/test | Calmar tune/test |
|---|---|---|---|
| Buy & hold | 17.9 / 16.8 | −50.6 / −38.3 | 0.35 / 0.44 |
| VIX W20 | 5.3 / 9.4 | −39.2 / −14.0 | 0.14 / 0.67 |
| Sector realized W100 (prior best) | 9.2 / 9.1 | −34.9 / −13.1 | 0.26 / **0.70** |
| Sector implied-where-avail W100 | 9.3 / 8.0 | −35.0 / −14.3 | 0.27 / 0.56 |
| Sector implied + short/long (20 OR 100) | 6.8 / 5.2 | **−22.9** / −13.7 | 0.30 / 0.38 |

- **Implied vs realized**: using OVX/GVZ/VXD (implied) for energy/materials/
  industrials instead of realized ETF vol was ≈neutral — realized was already fine.
- **Short + long windows** (risk-off above the 20-day OR the 100-day average):
  deepens drawdown protection (tune −35% → −23%; beats the prior best on drawdown
  for ~60% of stocks in both eras) but gives up more return (CAGR ~9% → ~6%) and
  does not improve test Calmar. A more-defensive dial, not a strict win.
- **Best risk-adjusted overall remains sector-realized W=100** (test Calmar 0.70).
  Use the short+long hybrid only if minimizing drawdown matters more than return.
