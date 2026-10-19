# Academic Trading — personal stock analysis tool (version 1)

A program that runs on your own Windows computer and opens in your web browser. For any US stock, it gives a **Buy**, **Sell** or **Hold** rating, built from the rules in your *Quantitative Trading Tool — Framework Design* (rev. Oct 16, 2026). It also shows the reasons for the rating, an exit plan, and a back-test of every rule.

---

## 1. One-time setup (about 5 minutes)

1. **Install Python** (free) from <https://www.python.org/downloads/>. Choose version 3.11 or newer.
   On the first installer screen, **tick "Add python.exe to PATH"**, then click *Install Now*.
2. **Unzip** this folder somewhere convenient, for example `Documents\academic_trading`.
3. **Double-click `setup_windows.bat`.** A black window opens and downloads the libraries the tool needs. When it says *Setup finished*, press any key.

## 2. Using the tool

- **Double-click `run_windows.bat`.** Your browser opens the tool at `http://localhost:8501`.
  Keep the black window open while you use the tool; close it to stop the tool.
- Type a ticker (for example `AAPL`) in the box on the left and press **Analyse**.
  Type `DEMO` to try it with simulated prices. This works even without an internet connection.
- The first analysis of a stock downloads 20 years of prices, which takes a few seconds.
  After that, the tool reuses the copy it saved that day. **Re-download** fetches fresh prices.
- Ratings use **closing prices**, so look after 4:00 pm New York time.

| Tab | What you see |
| --- | --- |
| Rating | Today's Buy / Sell / Hold, the three group scores, which rating rules pass, the trade plan (target, time limit, stop-loss), and every signal that counts today. |
| Chart | Candles, the 50- and 200-day averages, support/resistance zones, chart patterns with their targets, volume (coloured by the volume check) and RSI. |
| Signals | Every signal in a period, with the outcome of each pattern trade. You can download the list as a CSV file. |
| Back-test | Each rule compared with buy-and-hold after costs, for training, validation and final-test periods. Also pattern outcomes (target hit / time limit / stopped out). |
| Watch list | Rate a whole list of stocks at once. Also shows a cross-stock test of each rule (Bernoulli test, Framework 9 step 5). |
| Options data | Which sentiment systems are active, daily options snapshots, the Cboe archive download, and importing history files. |
| Data quality | What the cleaning step found and changed. |
| Settings | Weights, cut-offs, filters, costs and more. Values are saved in `data\settings.json`. |

## 3. Options sentiment data (Framework 6)

Free data covers the **VIX** (downloaded automatically). A history of each stock's own **put/call ratio** and **implied volatility** is not free, so the tool builds one, one day at a time:

- On the Watch list tab, press **Save today's options data for all**, once each trading day after 4:30 pm New York time. Or double-click `collect_options_windows.bat`.
- **To make Windows do this automatically:**
  1. Open the Start menu, type **Task Scheduler** and open it.
  2. Click **Create Basic Task…** and name it "Options snapshot". Choose **Daily**, start time **5:00 PM** (New York time; adjust for your time zone), then **Start a program**.
  3. For *Program/script*, browse to `collect_options_windows.bat` in this folder. Set *Start in* to this folder's path. Click Finish.
  4. Results are added to `data\collect_log.txt`.
- The stock-level systems switch on once enough days are saved: about 60 trading days for Tsunami and the implied-volatility system, and 110 for Squeeze Play II.
- If you buy history files (for example from Cboe DataShop or ORATS), import them on the Options data tab. Any CSV file with a date column and a value column works.
- **Equity-only put/call ratio** (Squeeze Play I): the tab has a button that downloads Cboe's free archive, which ends in 2019. Later years need an imported file.

## 4. What version 1 includes

| Framework | Included now |
| --- | --- |
| 1 Cleaning | Logical checks (impossible prices, closed-market days, negative volume, bad quotes), rolling 3-standard-deviation check on returns, straight-line filling of gaps, the 5%-missing-per-year rule, and a full change log. |
| 2 Turning points | Rolling window (strict inequalities, confirmed *w* days later) and PIPs (Euclidean, perpendicular and vertical distances). |
| 3 Trend | MA crossover 10/50, MACD 12/26/9, momentum 12/9, 20- and 50-day breakouts (with the "yesterday" condition), price vs. SMA 10/50/200, trend state, price / time / whole-day filters, and Alexander's filter as a benchmark only. |
| 4 Reversals | Support/resistance zones (corrected bin formula, hits, bounces, failures); head-and-shoulders and inverse, triple and double tops/bottoms, bull/bear flags and pennants, falling/rising wedges, rounding bottoms/tops (circle method); RSI (plain sums), Bollinger (book's standard deviation, with the usual version shown alongside), and ROC with adaptive levels. |
| 5 Volume | Relative-volume weights of 1, 0.5 and 0 (no adjustment for rounding patterns). |
| 6 Sentiment | 5 / 21 / 50 / 100 oscillators; Squeeze Play I, Squeeze Play II A and B, Tsunami, and the fast volatility swing; extreme readings measured against the record so far. |
| 7 Rating | Weights 0.35 / 0.40 / 0.25, the ±0.40 rule with the 2-of-3 and veto conditions, the strong-trend context check, the 2-day confirmation, and the reasons for every rating. |
| 8 Trade plan | Table 5.3 targets and time limits; stop after 2 closes back across the broken line or a 4% loss; entry at the next open; early close on an opposite signal. |
| 9 Validation | 60/20/20 time split; one-sided Welch t-tests against buy-and-hold (buys, sells, buy-minus-sell); 0.05% commission plus 0.05% half-spread per trade; hit rates; pattern outcomes; the Bernoulli test across stocks. |

**Planned for version 2:** the random-price bootstrap (AR(1) / GARCH-M / E-GARCH, with Ljung–Box and ARCH checks), counting patterns on random series, White's Reality Check or Hansen's SPA test, forecast-accuracy measures, all LMW items with the shadow ratings B and C, and the Dynamic Time Warping panel.

## 5. Choices made where the design leaves a gap

These are labelled "tool choice" in the code, so they are easy to find and change.

1. **Winsorizing changes the statistics, not the prices.** Extreme returns are pulled back to the 3-standard-deviation limit in a separate column, which the tests use. Chart prices, targets and stops stay at the real traded levels.
2. **Bad quotes** are recognised as a one-day move of more than 25% that is fully reversed the next day. A move of more than 40% that is *not* reversed is kept, but flagged for you to check.
3. **Patterns become actionable only after the last turning point is confirmed.** The signal day is the first close beyond the neckline (or trigger line) on or after that day, and before the book's deadline. This avoids using future prices. A test in `tests/` checks that cutting off the future never changes a past signal.
4. **Combining votes.** Each rule has at most one live vote (its most recent signal). The Reversal and Sentiment groups add up their weighted votes and cap the total at ±1. The Trend group averages its live votes, as the design says.
5. **Context check.** Against a strong trend, only a confirmed chart pattern can vote for a reversal. Zone bounces and RSI / Bollinger / ROC votes against the trend are shown but ignored.
6. **Filters.** The time filter (k = 1 day) is the default. Price and whole-day filters apply only to rules where they make sense; other rules fall back to the time filter.
7. **Stock implied volatility** is the plain average of the implied volatility of every option contract that traded that day.
8. **Back-test returns** run from the next day's open to the close *h* days after the signal, and the buy-and-hold benchmark is measured the same way.
9. **Triple tops/bottoms** require all three peaks (troughs) to be within 4% of each other. This combines the design's wording with the book's equation 5.17.

## 6. Checking the formulas

Double-click `run_tests_windows.bat` to run the checks. They compare the RSI, Bollinger, EMA, LWMA, bin and Bernoulli formulas with the book and the design. They also test pattern detection on drawn shapes, and check that no signal looks ahead.

## 7. Files

```
app.py                     the browser app
atrade/                    the analysis engine, one file per framework
  cleaning.py (F1)  turning_points.py (F2)  trend.py (F3)  zones.py, patterns.py, gauges.py (F4)
  volume.py (F5)    sentiment.py (F6)        rating.py (F7) trades.py (F8)  validation.py (F9)
  data.py  config.py  analysis.py  demo.py  collect.py
tests/test_core.py         formula and look-ahead checks
data/                      created on first run: saved prices, options snapshots, settings
```

*This is a research tool built from academic studies, not financial advice. Both books found that most technical rules did not beat buy-and-hold after costs.*
