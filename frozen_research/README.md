# frozen_research — frozen research scripts, NOT maintained

These are one-shot research scripts kept for the record. **They are frozen and
not maintained.** They are not part of the `atrade/` engine, nothing in the app
or the test suite imports them, and they are not expected to keep working as the
rest of the codebase changes.

Treat them as a lab notebook, not as code to run day to day:

- They have the repo path (`/home/user/academic_trading`) hard-coded, so they
  run only from that location and will need those paths edited to run elsewhere.
- They may lag behind the maintained modules they were derived from.
- Their conclusions — not their code — are the thing worth keeping. Each study's
  honest write-up lives in `results/sentiment_tuning/`.

## What lives here

- `phase3/` — the correlation-to-volatility study for market regimes and
  turning points. Findings: `results/sentiment_tuning/PHASE3_FINDINGS.md`.

## The maintained version

The useful part of the phase3 study was rewritten, cleaned up and wired into the
app as the information-only "Market stress / capitulation" card. That lives in
**`atrade/regime.py`** (called from `app.py`) and is covered by the normal test
suite. If you want the current market-stress reading, use that — not the scripts
here.
