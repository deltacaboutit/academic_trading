### What the tool does

Each time you analyse a stock, the tool runs the frameworks from your design document in order. It then gives a **Buy**, **Sell** or **Hold** rating, the reasons for it, and an exit plan.

| Stage | Framework | What happens |
| --- | --- | --- |
| Prepare the data | 1. Cleaning | Removes impossible prices, fills short gaps by straight-line interpolation, and flags extreme daily moves. |
| | 2. Turning points | Marks local peaks and troughs (rolling window and PIPs). A turning point is used only after it is confirmed, *w* days later. |
| Read the market | 3. Trend | Moving-average crossover, MACD, momentum, breakouts, price vs. 10/50/200-day averages, and an uptrend / downtrend / sideways label. |
| | 4. Reversals | Support and resistance zones; head-and-shoulders, triple and double tops/bottoms, flags, pennants, wedges, rounding tops/bottoms, and (added from PatternFinder 2.5) triangles, rectangles, cup and handle and island tops/bottoms; RSI, Bollinger Bands and adaptive ROC. |
| | 5. Volume | A signal on a day with heavy volume (at least 1.5× normal) counts fully. Normal volume counts half, and light volume counts zero. |
| | 6. Sentiment | Squeeze Play I and II, Tsunami and the fast volatility swing. These are contrarian: they buy when options traders are fearful. Until enough real volume history exists, they use put/call ratios from open interest (labelled "substitute"). |
| | [GEX] add-on | Gamma flip, call wall and put wall on the chart. The walls are the **strikes** with the biggest call and put positions (the usual industry definition). The flip is the price where market makers' hedging switches direction, found by re-pricing the options at many possible prices. Three GEX rules are back-tested, but GEX has no vote in the rating (it is not from the books). |
| Decide | 7. Rating | Score = 0.35 × Trend + 0.40 × Reversal + 0.25 × Sentiment. Buy needs a score of at least +0.40, 2 of 3 groups positive and no group below −0.50 (Sell is the mirror image). The rating changes only after 2 days in a row. |
| | 8. Trade plan | Every signal enters at the next day's open. Pattern trades exit at the price target, the time limit or a stop-loss (2 closes back across the broken line, or a 4% loss), whichever comes first. |
| Prove it | 9. Back-test | Tests every rule in time order (training, validation and final test), compares it with buy-and-hold after trading costs, and reports hit rates. |

### Words used in the app

The **Pattern guide** tab explains every chart pattern with a textbook example, and has a longer glossary. The Rating tab has a **summary in plain English**, and the Signals tab can **explain any pattern** it found.


- **Signal:** a rule saying "buy" or "sell" on a particular day.
- **Live:** a signal still counts in the rating, usually for 10 trading days, or until its pattern trade closes.
- **Forming pattern:** a chart pattern whose shape is complete but whose price has not yet closed beyond its line. It is shown on the Chart and Signals tabs as a heads-up only: it has **no vote** in the rating until it breaks out. It can then end as Failed (price went the wrong way first), Expired (no breakout in time) or Confirmed (it broke out and became a signal), and a confirmed pattern's trade ends at Target reached, Stopped out or Time limit reached.
- **Textbook checks:** 21 yes/no questions the classic chart books ask before trusting a breakout, such as "was volume heavy on the breakout day?" or "does the market agree?". They are shown for every chart pattern as information only and never change the rating.
- **Hit rate:** the share of signals after which the price moved the predicted way.
- **p-value:** the chance a result this good would appear by luck alone. Below 0.05 is the usual cut-off.
- **Random twins and edge:** each chart-pattern trade is compared with 200 trades of the same shape started on random days near it. The **edge** is how much better the real trade did. If the patterns had real skill, the edge would be clearly above zero. The All stocks tab shows the result over every stock.
- **Buy-and-hold:** simply owning the stock. Each rule must do better than this to be worth using.

### Things to keep in mind

- Both books found that most technical rules **did not** reliably beat buy-and-hold once costs and chance were taken into account. Support-zone bounces were the strongest finding, and even they did not beat buy-and-hold.
- Patterns also appear in random prices. Try the `DEMO` ticker: its prices are simulated, yet the tool still finds patterns and "signals" in them.
- Ratings use daily closing prices. Check them after 4:00 pm New York time.
- This is a research tool, not financial advice.

### How the rules are proven (version 2)

- **Validation tab:** a rule counts only if, in the final 20% of history, it beats buy-and-hold after costs **and**
  beats artificial price histories. These are built from the stock's own daily moves, shuffled so that any real pattern
  is destroyed, using the book's AR(1), GARCH-M and E-GARCH models. A model is used only if it fits.
- **The many-rules problem:** when dozens of rules are tried, one will look good by luck. White's Reality Check and
  Hansen's SPA test allow for that.
- **LMW tab:** a second, independent way of finding patterns (smoothed prices, ten patterns from Lo, Mamaysky and
  Wang 2000). It never changes the official rating. Shadow ratings B and C show what would happen if it did.
- **Similarity tab:** past stretches shaped like the last two months, and what happened next. Context only.
- **All stocks tab:** the same tests on every stock at once. A rule that works on one stock may be luck; a rule that
  works on most stocks is more convincing.
