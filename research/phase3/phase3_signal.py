"""Phase 3 model - print the CURRENT market reading (stress regime + bottom flag).

Uses the tuned correlation-to-volatility rules. This is a *coincident* stress /
bottom detector, not a forward-return predictor or a top-caller (see
results/sentiment_tuning/PHASE3_FINDINGS.md).

Bottom detector presets (all share vol-spike>1.3 and 20-day correlation>0.26; they
differ only in how deep a drawdown counts as a capitulation):
  major      drawdown < -18%  -> only major crashes, ~86% recall at ~100% precision
  correction drawdown < -10%  -> also ordinary ~10% pullbacks, ~91% recall / ~84% precision (2010-now)
The 20-day correlation window confirms during fast crashes (fixes the 2025-04-08 miss).

Run:  python research/phase3/phase3_signal.py            # shows both presets
      python research/phase3/phase3_signal.py correction # one preset
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, phase3

BEAR = dict(vspike_gt=1.15, corr_gt=0.22)                   # high-stress regime label
PRESETS = {
    "major":      dict(vspike_gt=1.3, corr20_gt=0.26, dd_lt=-0.18),
    "correction": dict(vspike_gt=1.3, corr20_gt=0.26, dd_lt=-0.10),
}


def is_bottom(row, p):
    return (row["vspike"] > p["vspike_gt"]) and (row["corr20"] > p["corr20_gt"]) and (row["dd"] < p["dd_lt"])


def main(which="both", lookback=12):
    names = list(PRESETS) if which == "both" else [which]
    df = phase3.market_features().dropna(subset=["vspike", "corr", "corr20", "dd", "ext"])
    r = df.iloc[-1]
    bear = (r["vspike"] > BEAR["vspike_gt"]) and (r["corr"] > BEAR["corr_gt"])
    print(f"Phase 3 market reading as of {df.index[-1].date()}")
    print(f"  VIX {r['vix']:.1f} | vol-spike (VIX/50d MA) {r['vspike']:.2f} | "
          f"correlation 60d {r['corr']:.2f} / 20d {r['corr20']:.2f} | drawdown {100*r['dd']:.1f}% | extension {100*r['ext']:+.1f}%")
    print(f"  stress regime : {'HIGH-STRESS (risk-off)' if bear else 'calm (risk-on)'}")
    for nm in names:
        flag = is_bottom(r, PRESETS[nm])
        print(f"  bottom flag [{nm:10s} dd<{int(100*PRESETS[nm]['dd_lt'])}%]: "
              f"{'CAPITULATION / possible bottom' if flag else 'no'}")
    print(f"  note: a coincident detector of stress/bottoms, NOT a buy/sell timer and NOT a top-caller.")
    hdr = " ".join(f"{nm[:4]+'?':>6s}" for nm in names)
    print(f"\n  last {lookback} sessions:")
    print(f"  {'date':12s} {'vix':>6s} {'vspike':>7s} {'corr20':>6s} {'dd%':>7s} {hdr}")
    for d, row in df.tail(lookback).iterrows():
        flags = " ".join(f"{('YES' if is_bottom(row, PRESETS[nm]) else '-'):>6s}" for nm in names)
        print(f"  {d.date().isoformat():12s} {row['vix']:6.1f} {row['vspike']:7.2f} {row['corr20']:6.2f} "
              f"{100*row['dd']:6.1f}% {flags}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    if arg not in ("both", *PRESETS):
        print(f"usage: phase3_signal.py [both|{'|'.join(PRESETS)}]"); sys.exit(1)
    main(arg)
