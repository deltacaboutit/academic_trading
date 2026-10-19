"""Save today's options snapshot for every stock on the watch list.

Run after the market closes:  python -m atrade.collect
(collect_options_windows.bat does this for you.)
"""
from __future__ import annotations

import datetime as dt
import sys

from .config import DATA_DIR, load_watchlist
from .data import snapshot_count, snapshot_options


def main(tickers=None):
    tickers = tickers or load_watchlist()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = DATA_DIR / "collect_log.txt"
    lines = [f"--- {dt.datetime.now():%Y-%m-%d %H:%M} ---"]
    for t in tickers:
        try:
            row = snapshot_options(t)
            lines.append(f"{t}: put/call {row['put_call_ratio']:.3f}, avg IV {row['avg_iv']:.3f}, "
                         f"{snapshot_count(t)} days saved")
        except Exception as e:
            lines.append(f"{t}: FAILED - {e}")
    text = "\n".join(lines)
    print(text)
    with open(log, "a", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or None)
