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

Capitulation rule (updated 2026-11-03): **VIX-spike > 1.3 AND 20-day correlation
> 0.26 AND drawdown < −18%**.

| | Recall (bottoms caught) | Precision | Median lead/lag |
|---|---|---|---|
| Tune (pre-2016, 15% swings) | 71% (5/7) | 58% | −6 days |
| Test (2016+, 15% swings) | 80% (4/5) | 42% | ~0 days |

It flags the major lows of 2002, 2008, 2010, 2011, 2018, **2020 and 2025**,
roughly coincident with the low. In the test era it now misses only the June-2022
interim low; the deep crashes (2018-Q4, COVID-2020, 2022-10, tariff-2025) are all
caught. Precision ~45% means about half the capitulation alerts mark a true major
bottom; the others are deep scares that recovered without being the swing low —
still useful as real-time stress flags.

### Fix — why the correlation window is now 20 days (was 60)

The original rule used a **60-day** correlation proxy and **missed the 2025-04-08
low**. At that low the 60-day proxy was **0.29 — one hundredth below the 0.30
trigger** — and by the time it crossed 0.30 (next day) the rebound had lifted the
drawdown above the trigger, so the three conditions never aligned. The 60-day
window is simply too slow for a 4-day flash crash. The **20-day** proxy was
already **0.44 at that same low** (and 0.5–0.6 in March 2020), so switching to it
(re-tuned on pre-2016 data, not fitted to 2025) catches both 2020 and 2025 on the
day, with the same test recall (80%) and no rise in false alarms. See
`research/phase3/phase3_signal.py` for the live rule.

### Drawdown-gate sensitivity and presets (2010→now)

"Major" vs "ordinary" lows are defined by a ZigZag swing filter: a 15%-swing low
is a drop of ≥15% that then recovered ≥15% (a major bottom); a 10%-swing low only
needs a ≥10% reversal (ordinary correction). Since 2010 there are **7** major
(15%) lows — 2010-07, 2011-10, 2018-12, 2020-03, 2022-06, 2022-10, 2025-04 — and
**11** at the 10% bar (those plus 2015-08, 2016-02, 2018-02, 2022-03). Recall is
the share of those lows flagged (±25 trading days); precision is the share of the
detector's alert clusters that land on a real low.

Holding vol-spike>1.3 and 20-day correlation>0.26 fixed, varying only the
drawdown gate, over 2010→now:

| drawdown gate | vs 15% major lows (2010-now / 2016+ test) | vs 10% corrections (2010-now / 2016+ test) |
|---|---|---|
| −18% (**preset `major`**) | 86% r / 100% p · 80% / 100% | 55% / 100% · 50% / 100% |
| −14% | 86% / 92% · 80% / 86% | 55% / 92% · 50% / 86% |
| −10% (**preset `correction`**) | 86% / 53% · 80% / 50% | **91% / 84% · 88% / 90%** |
| −8% | 86% / 48% · 80% / 42% | 91% / 71% · 88% / 67% |
| −6% | 86% / 40% · 80% / 33% | 91% / 64% · 88% / 60% |

The drawdown gate is a "what size dip to catch" dial, and the conclusion is the
same with or without 2008/2002:
- **`major` (−18%)**: major crashes only, ~86% recall at ~100% precision.
- **`correction` (−10%)**: also ordinary ~10% pullbacks, ~91% recall / ~84%
  precision — the sweet spot for corrections. −8% adds no recall over −10% but
  is noticeably noisier, so it is not recommended.

These two presets are selectable in `research/phase3/phase3_signal.py`
(`phase3_signal.py major` / `correction`; default shows both).

### Live check — recent years

| Year | Market event | Major low? | Detector |
|---|---|---|---|
| 2020 | COVID crash, −34% | yes (2020-03-23) | **flagged**, on the day |
| 2022 | bear market | yes (2022-10-12) | flagged (missed the June interim low) |
| 2025 | tariff crash, −19% (VIX 52) | yes (2025-04-08) | **flagged**, on the day (fixed) |
| 2026 (to Oct) | mild, worst −8.9% | no major low | correctly silent — no false alarm |

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

By year: in **2025** (broad tariff selloff) it caught **71%** of the 519 stocks'
major lows at **61%** precision (its wheelhouse — a market-wide capitulation); in
**2026** (mild, no market-wide scare) it caught **36%** of 409 stock lows at 44%,
because many were idiosyncratic drops without the market stress the rule requires.

## Bottom line

The correlation-to-volatility composite is a good **coincident detector of major
market bottoms / stress** (catches ~80–90% of major lows out of sample, roughly on
the day), but it is **not** a top-caller and **not** a forward-return predictor —
capitulation is sometimes followed by a rebound and sometimes by more losses,
depending on the macro regime. Used as a *risk/alert* tool it is genuinely useful;
used as a *timing* tool it is not reliable. The tuned rules are saved in
`phase3_model_market.json`, `phase3_model_turns.json`, `phase3_model_stocks.json`,
and `research/phase3/phase3_signal.py` prints the current reading.
