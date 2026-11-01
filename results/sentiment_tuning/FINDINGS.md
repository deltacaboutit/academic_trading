# Framework 6 (sentiment) tuning study — findings

*Last run: 2026-11-01. Code: `atrade/sentiment.py`, settings in `atrade/config.py`.*

## Question

Can the contrarian **Squeeze Play II-A** system (Summa, *Trading Against the
Crowd*) be tuned to beat buy-and-hold, without overfitting? The three knobs
tuned were the **sentiment entry threshold**, the **sentiment exit threshold**,
and the **price-trigger days**, set separately for long and short trades. The
stop-loss settings were left alone.

## Headline result

**No.** Out of 256 setting combinations per source, **not one beat buy-and-hold**
— in the tuning period *or* the held-out test period, for *either* sentiment
source. The study is sound: the backtest was first checked to reproduce the
original `squeeze_play_2a` entry and exit days exactly.

## Method

- **Strategy:** enter long on a fear extreme (oscillator above `+entry`) confirmed
  by a close above the prior `trig_long`-day high; mirror image for shorts; exit
  at the opposite extreme (`±exit`) or on the unchanged stops. Entry at next
  day's open; cost of 0.2% per round trip applied to every trade.
- **One shared set of settings across all ~516 S&P 500 stocks** (not tuned per
  stock).
- **Time split:** earliest 70% of history to tune, last 30% to test. The test
  part was never used for tuning.
- **Benchmark:** every result is compared to simply buying and holding the same
  stock over the same window.
- **Sentiment data.** The per-stock put/call and implied-volatility data (the
  book's intended input) is only ~5 months long, far too short to split, so the
  two long-history, market-wide sources were used instead:
  - **VIX** (market implied volatility): 2001–2026; tune 2001–2019, test 2019–2026.
  - **Cboe equity put/call ratio:** 2006–2019; tune 2006–2015, test 2015–2019.

## Recommended ("least-bad", most stable) settings

These sit in a broad plateau — neighbouring settings score almost the same, and
they rank near the top in *both* periods, so they are not a lucky single peak.
The exit threshold barely matters.

| Source | Entry | Exit | Long trigger | Short trigger |
|---|---|---|---|---|
| VIX | +10% | 2% | 1-day high | 10-day low |
| Put/call | +5% | 2% | 1-day high | 10-day low |

**Caveat:** the plateau's direction of improvement is "trade less and stop
shorting", i.e. the best setting is the one that most *resembles* buy-and-hold —
and it still loses. That is a degenerate optimum, not a sweet spot. Treat these
as the least-bad settings within a losing family, **not** a profitable strategy.

## Tuning period vs test period (recommended settings)

Average per-stock return, shown as plain percentages:

| Source | Period | Strategy | Buy & hold | % stocks beating B&H | Trades/stock |
|---|---|---|---|---|---|
| VIX | Tune 2001–2019 | ≈ −14% | ≈ +520% | 3.3% | 23.5 |
| VIX | Test 2019–2026 | ≈ +24% | ≈ +130% | 16.1% | 8.5 |
| Put/call | Tune 2006–2015 | ≈ +7% | ≈ +119% | 11.3% | 10.2 |
| Put/call | Test 2015–2019 | ≈ +14% | ≈ +63% | 18.0% | **1.0 (too few to trust)** |

**Is the tune→test gap a sign of overfitting? No.** The raw edge shrinks because
the test eras are shorter and less bull-dominated, but the test result is if
anything *slightly less bad* than the tune result (more stocks beat B&H, not
fewer), and the ranking of settings is stable. The strategy simply loses in both.

## Why it loses

- **Shorts lose money or break even.** Shorting on "the crowd is complacent"
  during a long rising market is a losing bet.
- **Longs do no better than just holding the same stock over the same days.** A
  matched-window comparison comes out roughly equal, and costs then make the
  strategy slightly worse. The sentiment timing adds no real value.
- **The strategy sits in cash most of the time**, missing the market's long-run
  rise that buy-and-hold captures.

## Per-stock view

The underperformance is broad, not one unlucky stock. On the test period only
**16–18% of stocks** beat buy-and-hold, and the top-10 stocks make the total
*more* negative. The stocks it "beats" are ones that fell (it won by avoiding
them); the ones it loses to worst are the big winners it missed (NVDA, AMD, PLTR).

## Risks and weaknesses

1. It does not beat buy-and-hold, in-sample or out-of-sample, on either source.
2. Market-wide sentiment means the signal is the same for all stocks, so the
   ~516 stocks are **not** independent tests — effectively a few macro bets.
3. Shorting costs (borrow fees) are not modelled; real shorts would be worse.
4. Few independent market regimes; one long bull market dominates.
5. The tuned optimum sits at the edge of the grid (trade as little as possible),
   a classic sign there is no genuine signal.
6. The per-stock put/call and IV data — the book's actual input — is only ~5
   months, so the real test of Summa's idea cannot be run yet.

## What this changed in the code

- `config.py`: split `sp2_extreme` into `sp2a_entry` / `sp2a_exit`; added
  `sp2a_trig_long` / `sp2a_trig_short` (was a hard-coded 3). Defaults reproduce
  the original behaviour exactly.
- `config.py`: added an extended II-A system on the VIX and the market put/call
  ratio, with the tuned values above and `sp2a_extended_enable` (**off by
  default**; when on it is back-tested only and does not vote in the live rating).
- `sentiment.py`: `squeeze_play_2a()` takes the four knobs; extended runs added.

## Files in this folder

- `all_combos_vix.csv`, `all_combos_pc.csv` — every one of the 256 combinations
  tried, with tune-period and test-period edge, for each source.
- `perstock_test_vix.csv`, `perstock_test_pc.csv` — per-stock test-period results
  for the recommended settings.

## Recommended next step

Market-wide VIX/put/call proxies are the wrong input. The only path that could
make this strategy worth trusting is to **collect per-stock put/call and
implied-volatility history going forward** (the daily collector already stores
it) and re-run this study once there are several years of it.
