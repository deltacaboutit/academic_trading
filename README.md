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

### Where the data comes from

Your **Watchtower** repo collects the Yahoo Finance option chain of every S&P 500 stock (plus SPY, QQQ and IWM)
each evening. The **Options data sync** GitHub Action in this repo copies each new Watchtower day across.
It runs at 11:15 pm and 7:15 am New York time and only processes days it hasn't seen.

### Where it is stored

| What | Where | Contents |
| --- | --- | --- |
| **Full chain, one file per stock** | `SPY.parquet`, `AAPL.parquet`, … on the GitHub Release **[options-data](https://github.com/deltacaboutit/academic_trading/releases/tag/options-data)**, grouped into 27 zip bundles by first letter (`chains_A.zip` …) because GitHub allows a workflow only 1,000 requests an hour. | Every contract on every day, appended: date, ticker, spot, expiry_str, strike, is_call, iv, iv_filled, oi, volume, T, bid, ask, last_price, source. |
| **Daily summary** | `options_data/summary/2026-10.parquet`, … (in git, one small file per month) | One row per stock per day: spot, put and call open interest and volume, put/call ratios from both, average IV, Watchtower's 30-day IV, total GEX, gamma flip, call wall and put wall. |
| Sync bookkeeping | `options_data/watchtower_state.json`, `universe.txt` | Which Watchtower days are done and how each was dated. |

**Why a Release and not the repo itself?** Git keeps a full copy of every version of every file. One file per stock
rewritten every evening would add the whole data set to the repo again each day, and it would hit GitHub's limits within
weeks. A Release asset is simply replaced, so storage stays at the real data size (about 5 MB of new data per day).
In Python, `atrade.options_store.read_chain("AAPL")` returns a stock's full chain and downloads the file if needed.

### Fixes applied on the way in

1. **Real trading day.** Watchtower names each file after the UTC date when its run *started*. So a run that starts
   after 8 pm New York time, or a weekend re-run, carries the next day's date. The sync dates each run by the New York
   date of its latest option trade instead. For example, Watchtower's file `2026-10-20` holds the Oct 19 session.
2. **Broken runs.** A run with less than half the usual number of contracts (for example Watchtower's `2026-08-28`, which
   has about 7,700 contracts instead of about 140,000) is skipped and logged. That day stays missing until you import it.
   If two runs belong to the same day, the more complete one is kept.
3. **Junk implied volatility.** Yahoo's 0.00001 floor and its 5.0 cap are replaced by straight-line interpolation between
   the nearest good strikes of the same expiry and type (call or put). Beyond the first or last good strike, the nearest
   good value is used. These rows have `iv_filled = True`.

### Adding missing days and older history

`python -m atrade.import_history` **only fills days a stock doesn't already have**. It accepts:

- `--watchtower-folder FOLDER`: missing days in Watchtower's own layout (`chains/ticker=XXX/date=….parquet`, and
  optionally `features/date=….parquet`).
- `--chain chain_history.parquet`, `--oi oi_history.parquet` and `--gex gex_history.csv`: your flat history files. The
  open-interest-only and GEX files reach back before Watchtower started (May and April).

**The easy way, all in the browser:**
1. On the repo's **Releases** page, open **Import inbox**, click the pencil (Edit), attach your files, and click *Update release*.
   Accepted files: `chain_history*.parquet`, `oi_history*.parquet`, `gex_history*.csv`, and `.zip` files of missing
   days in Watchtower's layout. Each file can be up to 2 GB.
2. On the **Actions** tab, choose **Options history import**, then *Run workflow*. When it finishes, the inbox is emptied.

The inbox appears after the first nightly sync run. On your own computer, the same import is
`python -m atrade.import_history --inbox FOLDER`.

### How the sentiment systems use the data

- Watchtower's chains include **real volume from June 2, 2026**. Days before that (your older files) have **open interest only**. Where there isn't yet enough real-volume history, the stock-level systems use **put open interest ÷ call open interest** instead. The same applies to Squeeze Play I, which uses a market-wide ratio built from all the S&P 500 stocks. These versions are labelled **"open-interest substitute"**.
- Once the real-volume version has enough history (60 days for Tsunami and Squeeze Play I, 110 for Squeeze Play II), it takes over the vote. The substitute keeps being back-tested alongside it, so you can compare the two.
- The stock's average implied volatility is the plain average over the contracts that traded that day, after the junk-IV fix. GEX levels come from Watchtower's own gamma engine, or from `gex_history.csv` for older days.

### GEX (gamma exposure) — an add-on, not from the books

The tool draws the **gamma flip, call wall and put wall** on the chart, and shows them on the Rating tab.
Three GEX rules are **back-tested only and have no vote in the rating**, as your framework requires for anything untested:

| Rule | Buy / Sell |
| --- | --- |
| [GEX] Put-wall hold | Buy: the low touched yesterday's put wall and the close stayed above it. |
| [GEX] Call-wall rejection | Sell: the high reached yesterday's call wall and the close stayed below it. |
| [GEX] Flip cross | Buy when the close crosses above yesterday's flip; sell when it crosses below. |

The rules use **yesterday's** levels, because today's levels are only known after today's close.

## 3b. Price history for testing (`price_data/`)

Double-click **`download_prices_windows.bat`**. It downloads 25 years of daily prices from Yahoo Finance for every
ticker in `options_data/universe.txt` (the S&P 500 stocks plus SPY, QQQ and IWM) and for the VIX. It takes
about 10–20 minutes. Tickers that fail are retried once automatically. At the end, it asks whether to upload the
`price_data` folder to GitHub.

| File | Contents |
| --- | --- |
| `price_data/<TICKER>.parquet` | One row per trading day. Open, High, Low and Close are adjusted for splits and dividends (what the tool analyses). Volume, `Close_raw` (the real closing price, which the option strikes refer to), Dividends and Splits. |
| `price_data/manifest.csv` | One line per ticker: first and last day, number of days, quick data checks, and any download error. |

The data serves as real test data, so the rules can be checked on real prices instead of simulated ones. The tool
also falls back to it if Yahoo can't be reached. **Refresh it rarely:** every full re-download rewrites all files,
and git keeps the old copies (about 100 MB each time). To fetch only tickers that are missing, run
`python -m atrade.download_prices --missing-only`.

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
  gex.py ([GEX] add-on)  options_store.py  watchtower.py (nightly sync)  release.py  import_history.py
  data.py  config.py  analysis.py  demo.py  collect.py
options_data/              daily options summary + sync bookkeeping (full chains are on the Release; see section 3)
.github/workflows/         options_sync.yml: the nightly copy from Watchtower
tests/                     formula, look-ahead, data-store and GEX checks
data/                      created on first run: saved prices and settings (not uploaded)
```

*This is a research tool built from academic studies, not financial advice. Both books found that most technical rules did not beat buy-and-hold after costs.*
