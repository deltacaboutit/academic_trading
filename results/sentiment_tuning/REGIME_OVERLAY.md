# Phase 1 — volatility regime overlay (cut drawdowns)

*Run 2026-11-02. Code: `scratchpad/regime.py`; results: `regime_overlay.csv`.*

## Idea

The Squeeze Play study showed that *swapping the volatility input never helps* —
the problem is the frequent, contrarian, long/short trade logic. So this phase
changes the **logic**: instead of trading in and out on short-term extremes, stay
invested in the stock most of the time and only **step aside to cash during a
high-risk (bear) regime** flagged by volatility. Long only (shorting already lost).

## Regime signal (market-wide, low-parameter, no look-ahead)

Risk-**off** when either:
- the **VIX is above its own W-day moving average** (a volatility spike), or
- the **VXV / VIX term structure inverts** below `k_term` (near-term fear above
  longer-term fear — a known stress signal).

The regime is read at the prior close and applied to the next day's return. Each
switch in or out of cash costs one one-way cost (0.1%). Only two knobs (`W`,
`k_term`) are tuned, to limit overfitting.

## Method

- Data: VIX and VXV (3-month VIX) from FRED, 2007-12 to 2026-10 (VXV's start).
- Time split with a crash in **each** half: tune < 2018 (2008, 2011, 2015), test
  ≥ 2018 (2018 Q4, 2020, 2022).
- Judged vs buy-and-hold on annual return (CAGR), worst drawdown (peak-to-trough
  loss), and Calmar (CAGR ÷ |drawdown|) — on an equal-weight market portfolio and
  across all ~500 stocks individually.

## Result — honest summary

**The overlay reliably reduces drawdowns, but pays for it with return.**

| Metric (equal-weight market) | Buy & hold | Overlay (best, stable) |
|---|---|---|
| Worst drawdown, tune (2008-era) | −50% | **−26%** |
| Worst drawdown, test (2020/2022) | −38% | **−14%** |
| Annual return, tune | +16% | +4–7% |
| Annual return, test | +17% | +8–10% |
| Fraction invested (rest in cash) | 100% | ~57–68% |
| Stocks with smaller drawdown | — | **77–85%** (both eras) |

- **Drawdown reduction is robust**: it holds in both periods and for ~80% of
  individual stocks — not a fluke of one crash.
- **Return is roughly halved** by the time spent in cash.
- **Risk-adjusted (Calmar) is NOT reliably improved.** Market Calmar: tune
  0.33 → 0.10–0.22 (overlay worse); test 0.45 → up to 0.70 for the best rule
  (overlay better). Because tune and test disagree, the Calmar gain is not
  dependable; only the drawdown reduction is.

## Recommended rule (for drawdown reduction, stable across periods)

**VIX above its 20-day average, OR VXV/VIX below 1.0.** (`W=20, k_term=1.00`.)
Cuts market drawdown to ~−27% (tune) / −14% (test) and reduces drawdown for
~82–84% of individual stocks, while staying invested ~57% of the time. Treat it
as a **risk dial**, not a return booster: it suits an investor who will accept
clearly lower returns to avoid big crashes. It does not beat buy-and-hold on
return, and it does not reliably beat it risk-adjusted.

## Caveats

- The regime is one market-wide signal applied to every stock, so the stock
  results are correlated (a few macro episodes drive everything).
- "Go to cash on a VIX spike" can sell after a drop has started and buy back
  higher (whipsaw); the drawdown benefit survives this, the return does not.
- Cash is assumed to earn 0% (no interest), which understates the overlay a little.
- Two knobs are still two knobs; the drawdown finding is robust, any Calmar edge
  is not.

## Next (Phase 3, separate)

A distinct composite "tops and bottoms" detector — judged on how well it flags
major market turns, not on a trading P&L. To be built separately from this overlay.
