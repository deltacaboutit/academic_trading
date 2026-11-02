# Sector volatility trackers — what's available

*Compiled 2026-11-02 (FRED + Yahoo probes).*

Goal: a per-sector volatility series to use as a sentiment / regime proxy, since
per-stock options data is only ~5 months long.

## Dedicated implied-volatility indices (CBOE "fear gauges")

These exist and are downloadable (FRED or Yahoo):

| Ticker | Covers | History |
|---|---|---|
| VIX | S&P 500 (broad) | 1990+ |
| VXN | Nasdaq-100 → **Technology / Communication Services** | 2001+ |
| VXO | S&P 100 (broad, old VIX) | 1986–2021 |
| VXD | Dow 30 → blue-chip / **Industrials-ish** | 1997+ |
| RVX | Russell 2000 → **small caps** (size, not a sector) | 2004+ |
| OVX | Crude oil → **Energy** proxy | 2007+ |
| GVZ | Gold → **Materials / precious metals** proxy | 2008+ |

No standalone implied-vol index exists (or is still published/downloadable) for
Financials, Health Care, Consumer Discretionary, Consumer Staples, Utilities, or
Real Estate. The old CBOE sector-ETF vol indices (e.g. VXXLE for energy) ran only
2011–2022 and are no longer downloadable.

## Realized-volatility trackers from sector ETFs (universal fallback)

Full price history is available for every SPDR sector ETF, so a realized-vol
tracker (e.g. annualised standard deviation of daily returns over a trailing
window) can be built for **all 11 GICS sectors**:

| ETF | Sector | History |
|---|---|---|
| XLK | Information Technology | 1998+ |
| XLC | Communication Services | 2018+ |
| XLF | Financials (KRE = regional banks, 2006+) | 1998+ |
| XLV | Health Care | 1998+ |
| XLY | Consumer Discretionary | 1998+ |
| XLP | Consumer Staples | 1998+ |
| XLI | Industrials | 1998+ |
| XLB | Materials | 1998+ |
| XLE | Energy | 1998+ |
| XLU | Utilities | 1998+ |
| XLRE | Real Estate | 2015+ |
| SMH | Semiconductors (sub-sector) | 2000+ |

## Recommended hybrid mapping

Use the implied-vol index where one fits the sector; otherwise use the ETF's
realized vol:

- Technology → VXN · Communication Services → VXN · Energy → OVX ·
  Materials → GVZ · Industrials → VXD · small caps → RVX
- Financials → realized XLF (or KRE) · Health Care → realized XLV ·
  Cons. Discretionary → realized XLY · Cons. Staples → realized XLP ·
  Utilities → realized XLU · Real Estate → realized XLRE

## Caveats

- Realized vol is backward-looking and is derived from the same stocks' prices,
  so as a *contrarian sentiment* input for a sector's own stocks it is partly
  circular. It is still a valid volatility-regime input.
- Earlier results stand: no volatility index (broad or sector/commodity) let the
  Squeeze Play II-A strategy beat buy-and-hold, and the regime overlay behaved
  the same across indices. Sector matching may refine the regime overlay a little
  but is unlikely to overturn those conclusions.
