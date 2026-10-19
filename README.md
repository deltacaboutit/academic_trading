# Academic Trading — personal stock analysis tool (version 1)

A program that runs on your own Windows computer and opens in your web browser. For any US stock, it gives a **Buy**, **Sell** or **Hold** rating, built from the rules in your *Quantitative Trading Tool — Framework Design* (rev. Oct 16, 2026). It also shows the reasons for the rating, an exit plan, and a back-test of every rule.

---

## 1. One-time setup (about 10 minutes)

1. **Install Python** (free) from <https://www.python.org/downloads/>. Choose version 3.11 or newer.
   On the first installer screen, **tick "Add python.exe to PATH"**, then click *Install Now*.
2. **Install Git** (free) from <https://git-scm.com/download/win>; the default choices are fine.
   Or use GitHub Desktop (<https://desktop.github.com>) if you prefer clicking to typing.
3. **Get the code.** In a Command Prompt, run these two lines:
   ```
   cd %USERPROFILE%\Documents
   git clone https://github.com/deltacaboutit/academic_trading.git
   ```
   In GitHub Desktop, use *File → Clone repository*.
4. **Double-click `setup_windows.bat`** in the new `academic_trading` folder. It downloads the libraries the tool needs.
   When it says *Setup finished*, press any key.

**To get the newest code and options data later**, double-click `update_windows.bat`, or press
*Get the latest data from GitHub* on the Options data tab.

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
| Chart | Candles, the 50- and 200-day averages, support/resistance zones, chart patterns with their targets, GEX levels, volume (coloured by the volume check) and RSI. |
| Signals | Every signal in a period, with the outcome of each pattern trade. You can download the list as a CSV file. |
| Back-test | Each rule compared with buy-and-hold after costs, for training, validation and final-test periods. Also pattern outcomes (target hit / time limit / stopped out). |
| Watch list | Rate a whole list of stocks at once. Also shows a cross-stock test of each rule (Bernoulli test, Framework 9 step 5). |
| Options data | Which sentiment systems are active (and whether each uses real volume or the open-interest substitute), the state of the options data store, a button to get the latest data, and importing extra history files. |
| Data quality | What the cleaning step found and changed. |
| Settings | Weights, cut-offs, filters, costs and more. Values are saved in `data\settings.json`. |

## 3. Options data (Framework 6) and GEX

### Where it lives

Everything is in the `options_data` folder of this repo:

| Path | What it holds |
| --- | --- |
| `summary/2026-10.parquet`, … | **Daily summary**: one row per stock per day, with one file per month. It holds spot price, put and call open interest, put/call ratio from open interest, put and call volume, put/call ratio from volume, average implied volatility, GEX flip, call wall and put wall. |
| `chain/part_001.parquet`, … | **Full contract-level chain** (every strike and expiry): date, ticker, spot, expiry_str, strike, is_call, iv, oi, T, volume, source. Each part is under 95 MB. |
| `chain/pending/2026-10-20.parquet`, … | The most recent days, one small file each, not yet merged into a part. |
| `universe.txt` | The tickers the daily update collects. |
| `manifest.json`, `update_log.csv`, `gex_calibration.csv` | What each part contains, a log of each daily run, and the GEX comparison from the import. |

The tool joins every part and pending file back into the full chain (`atrade.options_store.read_chain()`).

**Why the daily files?** Git keeps every version of every file. If one large part were rewritten each day, the repo
would grow by that part's full size every day and reach GitHub's limits within months. Instead, each day is written
once to its own small file. When the daily files add up to about 95 MB, they are merged into the next numbered part,
which is never changed again.

### One-time import of your existing history

1. Make a folder called `import` inside `academic_trading`. It is not uploaded to GitHub. Copy
   `chain_history.parquet`, `oi_history.parquet` and `gex_history.csv` into it.
2. Double-click **`import_options_windows.bat`**. The import does five things:
   - Writes the chain into parts.
   - Adds the open-interest-only history for the dates before the chain starts.
   - Builds the daily summary and puts your GEX levels into it.
   - Writes `universe.txt`.
   - **Calibrates the tool's GEX formula against your `gex_history.csv`**, so that new daily values continue your series. It prints how closely the walls and flip match.
3. Upload the result: in GitHub Desktop, commit `options_data` and press *Push*. Or in a Command Prompt:
   ```
   git add options_data
   git commit -m "Import options history"
   git push
   ```

### The daily update (GitHub Actions)

`.github/workflows/daily_options.yml` runs every weekday at **4:30 pm New York time**. GitHub's clock ignores
daylight saving, so two times are scheduled; the script checks the New York clock and the NYSE holiday calendar,
so only one run a day does any work. Each run does four things:

- Downloads every expiry for every ticker in `universe.txt` from Yahoo Finance: strike, call/put, implied volatility, open interest and **the day's real volume**.
- Saves the day as a new file in `chain/pending/` and adds the day's rows to the summary, with put/call ratios from both volume and open interest, average implied volatility and GEX levels.
- Merges the pending files into a new part once they add up to about 95 MB.
- Commits and pushes the new data.

It does nothing until the one-time import has created `universe.txt`. You can also start it by hand: go to the repo's
**Actions** tab → *Daily options update* → *Run workflow*.

Things to know:

- **Open interest** on Yahoo is the figure the OCC publishes overnight, so it reflects the **previous** trading day's close. Volume and spot are the current day's.
- Yahoo sometimes refuses requests from cloud servers. The script retries each ticker. The run fails (shown in red on the Actions tab) only if more than half of the tickers fail; `options_data/update_log.csv` records each run.
- If the push is ever refused, open the repo's *Settings → Actions → General* and set *Workflow permissions* to **Read and write**.

### How the sentiment systems use the data

- Your history has **open interest but no volume**. Until enough days of real volume exist, the stock-level systems use **put open interest ÷ call open interest**. The same applies to Squeeze Play I, which uses a market-wide ratio built from all the S&P 500 stocks. These versions are labelled **"open-interest substitute"**.
- Once the real-volume version has enough history (60 days for Tsunami and Squeeze Play I, 110 for Squeeze Play II), it takes over the vote. The substitute keeps being back-tested alongside it, so you can compare the two.
- The stock's average implied volatility comes from the chain. Junk values (IV below 1% or above 500%) are left out.

### GEX (gamma exposure) — an add-on, not from the books

The tool draws the **gamma flip, call wall and put wall** on the chart, and shows them on the Rating tab.
Three GEX rules are **back-tested only and have no vote in the rating**, as your framework requires for anything untested:

| Rule | Buy / Sell |
| --- | --- |
| [GEX] Put-wall hold | Buy: the low touched yesterday's put wall and the close stayed above it. |
| [GEX] Call-wall rejection | Sell: the high reached yesterday's call wall and the close stayed below it. |
| [GEX] Flip cross | Buy when the close crosses above yesterday's flip; sell when it crosses below. |

The rules use **yesterday's** levels, because today's levels are only known after today's close.

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
  gex.py ([GEX] add-on)  options_store.py  import_history.py  update_options.py
  data.py  config.py  analysis.py  demo.py  collect.py
options_data/              options history: daily summary + full chain parts (see section 3)
.github/workflows/         the 4:30 pm daily update
tests/                     formula, look-ahead, data-store and GEX checks
data/                      created on first run: saved prices and settings (not uploaded)
```

*This is a research tool built from academic studies, not financial advice. Both books found that most technical rules did not beat buy-and-hold after costs.*
