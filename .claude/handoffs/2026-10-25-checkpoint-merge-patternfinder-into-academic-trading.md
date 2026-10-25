# Handoff: Make academic_trading the main tool and retire PatternFinder2.5

**Mode**: Checkpoint
**Created**: 2026-10-25

## Where we are
The goal is to move every useful PatternFinder2.5 feature into academic_trading, so that PatternFinder2.5 can be retired. **Step 0 is done:** the 7 chart patterns missing from academic_trading have been added, tested and pushed (commit `c75128f`, branch `add-patternfinder-patterns`, no pull request yet). academic_trading now has 21 textbook-style patterns, plus the 10 shadow-only LMW patterns. Next, the remaining features need to be ported, in the priority order below.

**Both repos are needed:** `deltacaboutit/academic_trading` (the target) and `deltacaboutit/PatternFinder2.5` (the source). Attach both, and work on academic_trading.

## The user
- Not an expert in coding, with a basic understanding of statistics.
- Wants simple, full-sentence explanations, with jargon defined before it is used. This applies to replies, code comments, docs and in-app text.
- Uses Windows and runs everything through `.bat` files.
- More detail is in `PatternFinder2.5/CLAUDE.md`. academic_trading has no CLAUDE.md of its own.

## Decisions made this session
- **The new patterns follow academic_trading's own conventions, not PatternFinder2.5's.** That means its rolling-window turning points, the `known = idx + w` no-look-ahead rule, `Signal` records and the trade plan in `trades.py`, so the whole tool stays consistent.
- **Every threshold names its source or is marked "tool choice"** (in `atrade/config.py` and in the docstrings). Never tune a threshold on backtest results.
- **Island patterns need daily highs and lows.** `all_patterns()` takes optional `high` and `low` arguments. Without them, islands are skipped, so they are left out of the close-only random-price tests (Validation steps 3 and 4). This is documented in README choice 20.
- **The cup and handle got an extra "round, not V-shaped" rule (a tool choice).** The book's circle test let a sharp V-shape through. The rule requires at least half of the cup's closes to sit in the lowest 40% of its depth.
- **Rule names use academic_trading's sentence case** (for example "Cup and handle", not "Cup and Handle").

## Features still to port, in suggested order
Line numbers below are in **PatternFinder2.5** unless a path starts with `atrade/`.

1. **Daily-use features.** These make academic_trading a full replacement.
   - **"Forming" patterns and statuses.** PatternFinder2.5 shows patterns before their breakout, with the statuses Forming, Confirmed, Target reached, Stopped out, Failed and Expired: `pattern_core/patterns.py:47-53` and `evaluate()` at `:818`. academic_trading only creates a `Signal` once a breakout has happened, so a "forming" pattern needs a new kind of record that never votes in the rating.
   - **Plain-English pattern text and stock summary:** `pattern_core/report.py:90` (`pattern_text`) and `:188` (`summary_text`).
   - **Pattern Guide:** descriptions in `pattern_core/pattern_info.py:28` (`PATTERN_INFO`) and the glossary at `:226`. Pictures come from `pattern_core/pictures.py` and `pattern_core/examples.py`.
   - **Indicator sentences:** `pattern_core/indicators.py:134` (`describe_indicators`).
2. **Indicators and the confirmation checks.** These are what the user's pending v2.5 backtest tests.
   - **ATR and OBV:** `pattern_core/indicators.py:65` and `:84-104`. academic_trading's `atrade/indicators.py` has neither. ATR is the typical daily price range; OBV is a running total of volume on up and down days.
   - **The 21 textbook checks:** `pattern_core/confirm.py`. The `CHECKS` list is at `:69-136`, `NEEDS_BREAKOUT` at `:137`, `PROVEN` at `:142`, `key_reversal_days` at `:148`, `Context` at `:181`, `run_checks` at `:257` and `attach_checks` at `:505`. The market and relative-strength checks need SPY prices; academic_trading already stores those in `price_data/`.
3. **The research method.**
   - **Random twins:** `pattern_core/backtest.py:72` (`N_RANDOM = 200`) and `:178-213`.
   - **Realistic high/low fills:** `simulate()` at `:117`.
   - **The two trade-management variants:** the docstring at `:36-50`.
   - **Statistics:** in `pattern_core/analyze.py`, clustered standard errors at `:57`, Holm at `:125`, Benjamini–Hochberg at `:138`, `check_report` at `:217` and `variant_report` at `:270`.
   - **Keep academic_trading's existing tests.** These are the Reality Check/SPA, the artificial histories and buy-and-hold comparisons. Add the new tests next to them; don't replace anything.
4. **Data options.**
   - **Weekly and hourly bars:** `streamlit_app.py:31-32` and `pattern_core/data.py:52`.
   - **Loading a CSV price file:** `pattern_core/data.py:88`.
   - **Pattern-specific stop prices:** the `Scenario.stop` field in `pattern_core/patterns.py`. academic_trading uses one rule for every pattern: 2 closes back across the line, or a 4% loss (`atrade/trades.py:20`).

**Don't port:**
- the Tkinter desktop app and `.exe`, because the Streamlit app already covers this;
- `policy.py` and the Kronos notebooks, because the user abandoned that research;
- the ATR-based swing finder, because it is a different method, not a gap.

## Files in play (academic_trading)
- `atrade/patterns.py:439-680` — the 7 new detectors and `all_patterns(..., high, low)`. Done.
- `atrade/config.py:52-58` — the new pattern settings. Done.
- `atrade/analysis.py:80` — passes High and Low into `all_patterns`. Done.
- `tests/test_core.py:120` (the look-ahead test, which now includes highs and lows) and `:158-215` (the new pattern tests). Done.
- `atrade/signals.py` — the `Signal` record. Any new per-pattern data, such as checks or status, belongs in `extra`, or in new fields with defaults.
- `app.py:124` (tabs), `:314-340` (pattern drawing) and `:836` (Settings). Warning: `:329` formats `sg.target`, so every pattern needs a target.
- `atrade/validation.py:199` and `atrade/bootstrap.py:299` — the close-only callers of `all_patterns`.

## Repo state at handoff (academic_trading)
- **Branch**: `add-patternfinder-patterns` (pushed). `main` is unchanged.
- **Last 5 commits**:
  - `c75128f` Add seven chart patterns from PatternFinder 2.5
  - `a9821fa` Merge pull request #3 from deltacaboutit/chart-readability
  - `dc05224` Chart: show only the strongest indicator signals by default
  - `fea3911` Make the price chart easier to read
  - `5767112` Merge pull request #2 from deltacaboutit/streamlit-cloud
- **Uncommitted changes**: none, apart from this handoff file.
- **Tests**: `python -m pytest -q tests` gives 60 passed. Run it after every change.
- **PatternFinder2.5** is at `c0a6412`. Its upload wrongly includes `pattern_core/__pycache__/`. This is harmless, but should be removed before the repo is archived.

## Project docs worth re-reading
- `README.md`, section 5 (choices 1-20) — every "tool choice", including choice 20 for the new patterns. Add new choices here.
- `README.md`, "Version 2" table — add a row for each new feature.
- `HOW_IT_WORKS.md` — shown in the app's "How it works" tab, so keep it in sync.
- `PatternFinder2.5/docs/METHODOLOGY.md` — the source of every PatternFinder2.5 rule and threshold.
- `PatternFinder2.5/CLAUDE.md` — the PatternFinder2.5 rules: no look-ahead, fixed thresholds, statistics standards.

## Pending decisions (ask the user)
- **Pull request.** Should `add-patternfinder-patterns` be opened as a pull request and merged before more work is stacked on it?
- **Statuses for old patterns.** Should "Forming" patterns and statuses apply to the 14 original book patterns too, or only the 7 new ones? The recommendation is all of them.
- **Default stop rule.** Should pattern-specific stops replace the tool's 2-closes/4% stop, or be added as a trade-management variant? The recommendation is a variant, so the book's rule stays the default.
- **The v2.5 backtest.** The user's v2.5 backtest results haven't come back yet. Should PatternFinder2.5 stay usable until they do?
- **A known issue that isn't from this work.** The app logs a pyarrow warning ("Counts until" column) that was there before this work. Ask whether to fix it.

## Suggested first prompt for the next session
> Read `.claude/handoffs/2026-10-25-checkpoint-merge-patternfinder-into-academic-trading.md` in academic_trading, then the linked files in both repos. Ask me the pending decisions first. Then start step 1, "Forming" patterns and statuses: propose a design that fits academic_trading's `Signal` and no-look-ahead rules before writing code, and work on a new branch.
