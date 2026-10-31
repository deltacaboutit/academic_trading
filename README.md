# Academic Trading — personal stock analysis tool (version 1)

A program that runs on your own Windows computer and opens in your web browser. It can also be put online for free (section 2b). For any US stock, it gives a **Buy**, **Sell** or **Hold** rating, built from the rules in your *Quantitative Trading Tool — Framework Design* (rev. Oct 16, 2026). It also shows the reasons for the rating, an exit plan, and a back-test of every rule.

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
| Validation | Framework 9's full pass rule (steps 2, 3 and 7): which null models fit, the random-price test, patterns on random prices, the Reality Check / SPA, the accuracy of pattern price targets, and extra step-2 checks. **Run full validation** runs the random-price test for the stock on screen. |
| LMW | The [LMW] patterns found, their information tests, shadow ratings B and C next to the official rating, and a head-to-head with the textbook patterns. |
| Similarity | Past stretches shaped like the last two months, and what prices did next (Dynamic Time Warping). Context only. |
| Watch list | Rate a whole list of stocks at once. Also shows a cross-stock test of each rule (Bernoulli test, Framework 9 step 5). |
| All stocks | Results of the research run over every stock: cross-stock tests, pass-rule counts, LMW pooled tests (by exchange), rating variants, and the tuned rating weights. |
| Options data | Which sentiment systems are active (and whether each uses real volume or the open-interest substitute), the state of the options data store, a button to get the latest data, and importing extra history files. |
| Data quality | What the cleaning step found and changed. |
| Settings | Weights, cut-offs, filters, costs and more. Values are saved in `data\settings.json`. |

## 2b. Putting the tool online for free (Streamlit Community Cloud)

You can also run the tool as a website, so you can open it on any computer or phone without installing anything.
Streamlit Community Cloud hosts Streamlit apps like this one for free, straight from this GitHub repository.

**Deploy it (about 5 minutes, one time only):**

1. Go to <https://share.streamlit.io> and choose **Continue with GitHub**. Allow Streamlit to see your repositories.
2. Click **Create app**, then choose to deploy an app from GitHub.
3. Fill in the form:
   - **Repository:** `deltacaboutit/academic_trading`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** any name you like, for example `academic-trading`.
4. Open **Advanced settings** and choose **Python 3.12**. No secrets are needed.
5. Click **Deploy**. The first start installs the libraries and takes a few minutes. After that, the tool opens at
   your new web address.

**Good to know about the online version:**

- **It updates itself.** Every time GitHub changes (new code, or the daily options data), the online copy picks up the
  change within a few minutes. There is no *Get the latest data* button online, because it isn't needed.
- **Settings and the watch list are per visit.** Changes you make on the Settings or Watch list tab last until you
  close the page, and never affect other visitors. To change the starting values for everyone, edit
  `atrade/config.py` on GitHub.
- **Full validation is slower.** The free server is small and stops very long jobs, so *Run full validation* starts at
  100 artificial histories instead of 500, and its result is forgotten when the server restarts. For lasting results,
  use the *Research run (all stocks)* workflow on GitHub (see the All stocks tab).
- **Prices.** If the server cannot reach Yahoo Finance, the tool uses the prices saved in `price_data/` and says so
  under the stock's name.
- **Sleeping.** If nobody opens the app for a while, Streamlit puts it to sleep. The next visitor sees a button to
  wake it up, which takes about a minute.
- **Who can see it.** Anyone with the link can open a public app. You can limit it to chosen email addresses in the
  app's **Settings → Sharing** on share.streamlit.io.

## 3. Options data (Framework 6) and GEX

### Where the data comes from

The **Options data collection** GitHub Action (`options_collect.yml`) downloads the Yahoo Finance option chain of every
stock in `options_data/universe.txt` (the S&P 500 plus SPY, QQQ and IWM) after each trading day's close. It collects
what Watchtower collected, in the same way:
- every expiry up to 90 days out,
- every call and put with open interest of at least 5,
- the strike, implied volatility, open interest, volume, bid, ask, last price and the day's closing price.

**When it runs.** The main run is at about 4:45 pm New York time. Catch-up runs follow at about 9:30 pm and 7:15 am the
next morning. Each run only fetches stocks that don't have the day yet, so after a good main run the catch-ups finish in
a minute. A stock that fails is retried three times, then once more at the end of the run, and then by the catch-ups.
Only the 7:15 am run shows a red X, and only if fewer than 95% of stocks still have the day. The file
`options_data/collect_state.json` lists each day's count and any stocks that failed.

It never collects between 9:25 am and 4:15 pm on a trading day, because Yahoo's chains then show the live,
unfinished day. **Run it by hand:** Actions → *Options data collection* → *Run workflow*. You can give a few tickers, and choose
*force* to collect them again.

Days up to Oct 19, 2026 were copied from your Watchtower repo. The **Watchtower backup sync (temporary)** workflow
still copies any day the collector missed completely, and never touches a day the collector has. Delete
`options_sync.yml` when Watchtower is taken down.

### Where it is stored

| What | Where | Contents |
| --- | --- | --- |
| **Full chain, one file per stock** | `SPY.parquet`, `AAPL.parquet`, … on the GitHub Release **[options-data](https://github.com/deltacaboutit/academic_trading/releases/tag/options-data)**, grouped into 27 zip bundles by first letter (`chains_A.zip` …) because GitHub allows a workflow only 1,000 requests an hour. | Every contract on every day, appended: date, ticker, spot, expiry_str, strike, is_call, iv, iv_filled, oi, volume, T, bid, ask, last_price, source. |
| **Daily summary** | `options_data/summary/2026-10.parquet`, … (in git, one small file per month) | One row per stock per day: spot, put and call open interest and volume, put/call ratios from both, average IV, 30-day IV index (`ivx_30d`), total GEX, gamma flip, call wall and put wall. |
| Bookkeeping | `options_data/collect_state.json`, `universe.txt`, `rate.json`, `watchtower_state.json` | Each collected day's count and failures, the stock list, the latest 3-month T-bill rate, and which Watchtower days were copied. |

**Why a Release and not the repo itself?** Git keeps a full copy of every version of every file. One file per stock
rewritten every evening would add the whole data set to the repo again each day, and it would hit GitHub's limits within
weeks. A Release asset is simply replaced, so storage stays at the real data size (about 5 MB of new data per day).
In Python, `atrade.options_store.read_chain("AAPL")` returns a stock's full chain and downloads the file if needed.

### Fixes applied on the way in

The collector dates each day by the NYSE session whose close has passed. It measures time to expiry from that
session's 4:00 pm close, so the result doesn't depend on when the run happens. Fixes 1 and 2 below were for the
days copied from Watchtower; fix 3 applies to every day.

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
- The stock's average implied volatility is the plain average over the contracts that traded that day, after the junk-IV fix.
- The **30-day IV index** (`ivx_30d`, in percent) is a cleaner measure of the stock's implied volatility, using Watchtower's method
  (the Livevol method): take the two expiries either side of 30 days; for each, average the implied volatility of the 4 calls and
  4 puts with strikes nearest the price, weighting each by its vega (how much its price reacts to a change in implied volatility);
  then blend the two expiries by the square root of their days to expiry. Watchtower's own value is kept on the days copied from
  it; the daily collector calculates it from the chain on every other day. On the days where both exist, the two agree to within
  a few hundredths of a point. GEX levels: see *How the GEX levels are calculated* below.

### GEX (gamma exposure) — an add-on, not from the books

The tool draws the **gamma flip, call wall and put wall** on the chart, and shows them on the Rating tab.
Three GEX rules are **back-tested only and have no vote in the rating**, as your framework requires for anything untested:

| Rule | Buy / Sell |
| --- | --- |
| [GEX] Put-wall hold | Buy: the low touched yesterday's put wall and the close stayed above it. |
| [GEX] Call-wall rejection | Sell: the high reached yesterday's call wall and the close stayed below it. |
| [GEX] Flip cross | Buy when the close crosses above yesterday's flip; sell when it crosses below. |

The rules use **yesterday's** levels, because today's levels are only known after today's close.

#### How the GEX levels are calculated (mixed method, decided 2026-10-21)

GEX (gamma exposure) estimates how much hedging options market makers must do as the price moves. It is calculated
from every contract with an expiry up to 90 days out and open interest of at least 5:

    GEX of a contract = gamma × open interest × 100 × price² × 0.01      (dollars of hedging per 1% move)

Gamma comes from the Black-Scholes formula, using the contract's implied volatility and the 3-month T-bill rate from FRED.
Calls count as positive and puts as negative, because market makers are assumed to own the calls and owe the puts.
That assumption is the usual one, and it is also the main weakness of every GEX number.

| Level | Method | Why |
| --- | --- | --- |
| **Call wall** | The **strike** whose calls have the largest total GEX at today's price. | The industry definition. SpotGamma: walls are "the strike with the largest net call / put gamma". A wall is where the open interest actually sits, and you can check it against the chain. |
| **Put wall** | The **strike** whose puts have the most negative total GEX at today's price. | As above. |
| **Gamma flip** | Re-price the whole chain at 200 hypothetical prices around today's price. The flip is where total GEX changes from positive to negative. If small wiggles cause several sign changes, the last *stable* one is used (at least 5 grid points on each side). | A flip is a price level, not a strike, so it can only be found this way. This is Watchtower's method, unchanged. |
| **Total GEX** | The value of that price curve at today's price. | Watchtower's method, unchanged. |

**What changed.** Watchtower also took the walls from the price curve: the curve's highest point above the flip and its
lowest point below it. That gives levels between strikes (for example 334.95). It also nets calls against puts, and it
blurs neighbouring strikes together.

On 2026-10-21 the walls of all 80 past days (June 2 to Oct 19, 41,116 stock-days) were recalculated at strikes from the
stored chains. The column `wall_method = "strike"` marks this. The flip and total GEX were left as they were.

**What to expect.** Strike walls are often close to today's price, because gamma is largest for strikes near the
current price. On the past days, about a quarter of call walls were below the price and a quarter of put walls were
above it. The old walls almost never were, because they were forced to lie above or below the flip. A call wall below
the price means the largest call position is already in the money, which is useful information in itself.

The code is in `atrade/gex.py`: `daily_levels` for the levels and `strike_walls` for the walls. Recalculating every
past day again is `python -m atrade.collect_chains --recompute-walls`.

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

## 4. What the tool includes

### Version 1

| Framework | Included now |
| --- | --- |
| 1 Cleaning | Logical checks (impossible prices, closed-market days, negative volume, bad quotes), rolling 3-standard-deviation check on returns, straight-line filling of gaps, the 5%-missing-per-year rule, and a full change log. |
| 2 Turning points | Rolling window (strict inequalities, confirmed *w* days later) and PIPs (Euclidean, perpendicular and vertical distances). |
| 3 Trend | MA crossover 10/50, MACD 12/26/9, momentum 12/9, 20- and 50-day breakouts (with the "yesterday" condition), price vs. SMA 10/50/200, trend state, price / time / whole-day filters, and Alexander's filter as a benchmark only. |
| 4 Reversals | Support/resistance zones (corrected bin formula, hits, bounces, failures); head-and-shoulders and inverse, triple and double tops/bottoms, bull/bear flags and pennants, falling/rising wedges, rounding bottoms/tops (circle method), plus the seven patterns added later (see Version 2, row 4b); RSI (plain sums), Bollinger (book's standard deviation, with the usual version shown alongside), and ROC with adaptive levels. |
| 5 Volume | Relative-volume weights of 1, 0.5 and 0 (no adjustment for rounding patterns). |
| 6 Sentiment | 5 / 21 / 50 / 100 oscillators; Squeeze Play I, Squeeze Play II A and B, Tsunami, and the fast volatility swing; extreme readings measured against the record so far. |
| 7 Rating | Weights 0.35 / 0.40 / 0.25, the ±0.40 rule with the 2-of-3 and veto conditions, the strong-trend context check, the 2-day confirmation, and the reasons for every rating. |
| 8 Trade plan | Table 5.3 targets and time limits; stop after 2 closes back across the broken line or a 4% loss; entry at the next open; early close on an opposite signal. |
| 9 Validation | 60/20/20 time split; one-sided Welch t-tests against buy-and-hold (buys, sells, buy-minus-sell); 0.05% commission plus 0.05% half-spread per trade; hit rates; pattern outcomes; the Bernoulli test across stocks. |

### Version 2

| Framework | Added |
| --- | --- |
| 9 step 2 | Jarque–Bera check of the bell-curve assumption, a one-sided Mann–Whitney test that does not need it, and the t-test repeated on simple returns. |
| 9 step 3 | Random-price test: AR(1) (OLS), GARCH-M and E-GARCH (maximum likelihood, book eqs. 8.8–8.14); a model is used only if its key coefficients are significant and its leftover shocks pass the Ljung–Box and ARCH tests (20 lags, 10%); 500 artificial histories per model built from the stock's own resampled shocks; p-values for the mean and for volatility. A stock with no fitting model is "not testable". |
| 9 step 4 | Every chart pattern counted on the real prices and on 10 random series with the same start, length, drift and volatility, with a test of whether price targets are hit more often on the real prices. |
| 9 step 6 | White's Reality Check and Hansen's SPA test over all rules (stationary bootstrap). |
| 9 step 8 | MSE, RMSE, MAE, MAPE, NRMSE, NPRMSE, Theil's U1 and U2, POCID, IPOCID, POS and the regression check for the pattern price targets. |
| 9 pass rule | Steps 2, 3 and 7 together, in the final test. |
| 7 tuning | Weights and cut-off searched on the training period of all stocks, checked once on validation, final test reported as-is; switch them on in Settings. |
| 2 / 4e [LMW] | Kernel-regression turning points (38-day windows, 3-day lag), the ten LMW patterns, volume trend, information-only mode until the 9b tests give a direction. |
| 9b [LMW] | Normalised returns, decile chi-square and Kolmogorov–Smirnov tests (pooled, by NYSE/AMEX and Nasdaq), head-to-head with the textbook items, shadow ratings B and C. |
| 4d | Dynamic Time Warping similarity panel: Savitzky–Golay smoothing, derivative and subsequence DTW, threshold 15, 20-day skip, optional Sakoe–Chiba band. |
| 4b (added) | Seven chart patterns brought over from PatternFinder 2.5, which are not in the book: ascending, descending and symmetrical triangles, rectangles, cup and handle, and island tops and bottoms. They use the same turning points, no-look-ahead rule and trade plan as the book's patterns. Their rules come from Edwards & Magee, Murphy, O'Neil, Bulkowski and Lo, Mamaysky & Wang (2000); see choice 20 below. |

### The research run (all stocks)

On GitHub: **Actions → Research run (all stocks) → Run workflow**. It works in three stages:
1. Sets the LMW smoothing width and directions using the training data only.
2. Runs the per-stock tests in 20 parallel jobs.
3. Combines the results and tunes the rating weights.

It takes about 1–2 hours and commits everything to `results/`. To see the results in the app,
press **Get the latest data from GitHub** on the Options data tab, then open the **All stocks** tab.
For the NYSE/AMEX vs Nasdaq split, run `download_info_windows.bat` once first.

On one computer: `python -m atrade.research all --tickers AAPL MSFT SPY --B 100`.

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
10. **Null models "fit"** when their key coefficients are significant at 5%: ρ for AR(1); a1 and β for GARCH-M; G and A for E-GARCH. All t-statistics are shown.
11. **Step 3 passes** only if the rule beats the random histories (p < 0.05) under *every* null model that fits. Its returns are close-to-close on both real and artificial prices, because artificial histories have no opening prices. Sentiment and GEX rules can't be simulated, so their verdict notes "step 3 not applicable".
12. **LMW smoothing width.** The paper's cross-validation recipe doesn't produce a usable width on daily prices: the usual form picks almost none, and equation 13 as printed picks the widest allowed. The paper also doesn't publish the width it used. So the width is fixed so that patterns are found as often as LMW report (about 8.3 per stock-year), chosen on training data. It is 1.8 days by default.
13. **LMW double tops/bottoms** use the first turning point in the window as E1. The volume trend uses volume, because shares outstanding barely change within 38 days, so they cancel out of the turnover ratio.
14. **LMW directions.** A pattern gets a vote only if the pooled training-data tests find it informative; the vote follows the direction its returns lean. Until the research run has set them, HS/IHS/DTOP/DBOT use their usual directions in the shadow ratings and the rest are information-only. A directed pattern votes when the close breaks its outer boundary within 20 days.
15. **DTW threshold.** The book's threshold of 15 comes from its S&P 500 example; real two-month queries often have no match that close. So the panel also lists the 10 closest non-overlapping matches, marked "closest".
16. **Forecast accuracy** divides prices by the break-day close, so patterns from different stocks and years can be pooled.
17. **Reality Check / SPA** use each rule's daily return over buy-and-hold (positions held for the judging period, costs on changes), close to close.
18. **Weight tuning** maximises the average, across stocks, of each rating trade's return after costs minus buy-and-hold, on the training period. The weights step by 0.1 and the cut-off is chosen from 0.3, 0.4 and 0.5.
19. **Broken artificial histories are redrawn.** For a stock with a huge one-day crash and very persistent volatility (for example FISV under GARCH-M), an artificial history can occasionally spiral until its price reaches zero or infinity. That history is thrown away and a new one drawn, and the count is shown on the Validation tab. The research run uses only `price_data/` (including `^VIX`), and a stock that still fails is listed under "Stocks that failed" rather than stopping its batch.
20. **The seven added patterns** (from PatternFinder 2.5; the book has no rules for them, so each rule names its source in `atrade/patterns.py`):
    - **Triangles and rectangles** use five alternating turning points, as in Lo, Mamaysky & Wang (2000), with a window of 7 days (the same as wedges). A side counts as "flat" when its points are within 4% of each other, the same tolerance as triple tops. Ascending triangles may only break up, and descending triangles only down. Symmetrical triangles and rectangles may break either way. A triangle must break between half and three-quarters of the way to its apex, the point where its two lines meet (Edwards & Magee). The target is the pattern's height at its widest part, measured from the broken line.
    - **Cup and handle** (O'Neil): two rims within 4% of each other, 7 to 65 weeks apart; a cup 12–50% deep that passes the book's circle test, with at least half of its closes in the lowest 40% of its depth (this rejects V shapes); and a handle that stays in the cup's upper half. The target is the rim plus the cup's depth.
    - **Island tops and bottoms** (Murphy): a gap to a new 20-day closing high (or low), then, within 10 days, a gap the other way that leaves the days in between stranded. The target is the island's own height, and the time limit is the usual 10 days. Islands need each day's high and low, so they are **left out of the random-price tests** (steps 3 and 4), which only have closing prices.

## 6. Checking the formulas

Double-click `run_tests_windows.bat` to run the checks. They compare the RSI, Bollinger, EMA, LWMA, bin and Bernoulli formulas with the book and the design. They also test pattern detection on drawn shapes, and check that no signal looks ahead.

## 7. Files

```
app.py                     the browser app
atrade/                    the analysis engine, one file per framework
  cleaning.py (F1)  turning_points.py (F2)  trend.py (F3)  zones.py, patterns.py, gauges.py (F4)
  volume.py (F5)    sentiment.py (F6)        rating.py (F7) trades.py (F8)  validation.py (F9)
  gex.py ([GEX] add-on)  options_store.py  collect_chains.py (daily collection)  watchtower.py (temporary backup sync)  release.py  import_history.py
  bootstrap.py (F9 step 3)  lmw.py ([LMW])  similarity.py (F4d)  research.py (all-stock run)  results.py
  data.py  config.py  analysis.py  demo.py  collect.py  download_prices.py
options_data/              daily options summary + sync bookkeeping (full chains are on the Release; see section 3)
.github/workflows/         options_collect.yml (daily collection), options_sync.yml (temporary Watchtower backup), options_import.yml, research.yml
results/                   research results (bootstrap per stock, summaries, tuned weights)
price_data/                25 years of daily prices per stock (test and validation data)
tests/                     formula, look-ahead, data-store and GEX checks
data/                      created on first run: saved prices and settings (not uploaded)
```

*This is a research tool built from academic studies, not financial advice. Both books found that most technical rules did not beat buy-and-hold after costs.*
