# Phase 3 — correlation-to-volatility model for regimes and turns

*Run 2026-11-03. Code: `research/phase3/`. Models: `phase3_model_*.json`.*

Phase 3 uses a **different indicator from Phase 1**. Phase 1 timed a cash overlay
off the volatility *level* (VIX vs its moving average). Phase 3's core indicator
is the **correlation dimension**:

- **CORR** — a realized-correlation proxy = Var(index returns) ÷ mean Var(stock
  returns) over 60 days. It rises when stocks move together (systemic stress).
- combined with **VOL** (VIX vs its 50-day average), **DD** (trailing drawdown
  from the 252-day peak) and **EXT** (price vs its 200-day average).

All features are causal. Ground-truth turns come from a ZigZag swing filter used
for scoring only. Tuned on data before 2016, tested after.

> First check: the raw return-vs-VIX correlation is **not** useful on its own —
> it sits near −0.8 at both tops and bottoms (the ever-present leverage effect),
> so Phase 3 uses the cross-sectional correlation proxy instead.

## 1. Market bottoms — WORKS (the main success)

Capitulation rule: **VIX-spike > 1.1 AND correlation > 0.30 AND drawdown < −16%**.

| | Recall (bottoms caught) | Precision | Median lead/lag |
|---|---|---|---|
| Tune (pre-2016, 15% swings) | 86% (6/7) | 55% | −3 days |
| Test (2016+, 15% swings) | 80% (4/5) | 45% | −1 day |
| Test (2016+, 10% swings) | 88% (7/8) | 33% | ~0 days |

It flagged the major lows of 2002, 2008, 2010, 2011, 2018, 2020 and 2022, roughly
coincident with the low (within a few days). It missed the March-2009 secondary
low (vol had already receded from the Nov-2008 peak) and the April-2025 low at the
15% threshold. Precision ~45% means about half the capitulation alerts mark a true
major bottom; the others are deep scares that recovered without being the swing low
— still useful as real-time stress flags.

## 2. Market tops — DOES NOT WORK

Complacency rule (low vol + low correlation + extension): **recall ~0–20%,
precision <10%**. Tops form in calm markets, so the "complacency" state persists
for months or years during bull runs without marking the actual peak. Volatility
and correlation simply do not spike at tops. This is a genuine limitation, not a
tuning failure.

## 3. Directional regime / forward returns — DOES NOT GENERALISE

Labelling days bear (high vol + high correlation) vs bull and looking at forward
returns:

| | Bear next-20d return | Bull next-20d return |
|---|---|---|
| Tune (pre-2016) | −0.3% | +0.6% (bear worse, as hoped) |
| Test (2016+) | **+2.5%** | +1.0% (bear *better* — reversed) |

Out of sample, high-stress days were followed by **higher** returns, because by
the time volatility and correlation spike the fall has already happened and a
rebound usually follows. The composite tells you that you are *in* a stressed /
bottoming state — it does **not** predict forward direction.

## 4. Individual-stock capitulation — REGIME-DEPENDENT, NOT RELIABLE

Per-stock rule (own vol-spike + deep drawdown + market stressed), forward return
after a flag vs the unconditional average, pooled across stocks:

| | After-flag next-60d | Baseline next-60d |
|---|---|---|
| Tune (2001–2015) | −0.2% | +2.4% (worse — falling knives) |
| Test (2016+) | +14.3% | +3.1% (much better — rebounds) |

The sign **flips** between eras: catching capitulating stocks worked brilliantly
in the fast-recovery 2016+ markets and badly in the grinding 2008 bear. Not a
dependable edge.

### 4b. Individual-stock bottom DETECTION — modest but consistent

Reframed as *detection* (does the flag land on each stock's own major lows, by
ZigZag 20%?) rather than forward-return prediction, a single shared rule (stock
vol-spike > 1.4, drawdown < −20%, market stressed) is stable across eras:

| | Recall (lows caught) | Precision | Median lead/lag |
|---|---|---|---|
| Tune (pre-2016) | 49% | 53% | −1 day |
| Test (2016+) | 52% | 54% | 0 days |

So ~half of each stock's major lows are flagged as capitulation, coincident with
the low, and ~half the alerts mark a true major low — consistent in and out of
sample. It catches market-driven stock bottoms and misses idiosyncratic ones
(the rule requires market stress). Useful as a stock-level stress/bottom flag,
with the same caveat as the market: it is coincident detection, not a forward
predictor. Rule saved in `phase3_model_stock_turns.json`.

## Bottom line

The correlation-to-volatility composite is a good **coincident detector of major
market bottoms / stress** (catches ~80–90% of major lows out of sample, roughly on
the day), but it is **not** a top-caller and **not** a forward-return predictor —
capitulation is sometimes followed by a rebound and sometimes by more losses,
depending on the macro regime. Used as a *risk/alert* tool it is genuinely useful;
used as a *timing* tool it is not reliable. The tuned rules are saved in
`phase3_model_market.json`, `phase3_model_turns.json`, `phase3_model_stocks.json`,
and `research/phase3/phase3_signal.py` prints the current reading.
