"""Phase 3 model - print the CURRENT market reading (regime + bottom flag).

Uses the tuned correlation-to-volatility rules. This is a *coincident* stress /
bottom detector, not a forward-return predictor or a top-caller (see
results/sentiment_tuning/PHASE3_FINDINGS.md). Run:  python research/phase3/phase3_signal.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, phase3

# tuned rules (see PHASE3_FINDINGS.md). Bottom uses the FAST 20-day correlation
# proxy (corr20) so it confirms during fast crashes (fixes the 2025-04-08 miss).
BEAR = dict(vspike_gt=1.15, corr_gt=0.22)                   # high-stress regime label
BOTTOM = dict(vspike_gt=1.3, corr20_gt=0.26, dd_lt=-0.18)   # major-bottom capitulation


def main(lookback=10):
    df = phase3.market_features().dropna(subset=["vspike", "corr", "corr20", "dd", "ext"])
    r = df.iloc[-1]
    bear = (r["vspike"] > BEAR["vspike_gt"]) and (r["corr"] > BEAR["corr_gt"])
    bottom = (r["vspike"] > BOTTOM["vspike_gt"]) and (r["corr20"] > BOTTOM["corr20_gt"]) and (r["dd"] < BOTTOM["dd_lt"])
    print(f"Phase 3 market reading as of {df.index[-1].date()}")
    print(f"  VIX {r['vix']:.1f} | vol-spike (VIX/50d MA) {r['vspike']:.2f} | "
          f"correlation 60d {r['corr']:.2f} / 20d {r['corr20']:.2f} | drawdown {100*r['dd']:.1f}% | extension {100*r['ext']:+.1f}%")
    print(f"  stress regime : {'HIGH-STRESS (risk-off)' if bear else 'calm (risk-on)'}")
    print(f"  bottom flag   : {'CAPITULATION / possible major bottom' if bottom else 'no'}")
    print(f"  note: a coincident detector of stress/bottoms, NOT a buy/sell timer and NOT a top-caller.")
    print(f"\n  last {lookback} sessions:")
    print(f"  {'date':12s} {'vix':>6s} {'vspike':>7s} {'corr20':>6s} {'dd%':>7s} {'bottom?':>8s}")
    for d, row in df.tail(lookback).iterrows():
        bf = (row["vspike"]>BOTTOM["vspike_gt"]) and (row["corr20"]>BOTTOM["corr20_gt"]) and (row["dd"]<BOTTOM["dd_lt"])
        print(f"  {d.date().isoformat():12s} {row['vix']:6.1f} {row['vspike']:7.2f} {row['corr20']:6.2f} "
              f"{100*row['dd']:6.1f}% {'YES' if bf else '-':>8s}")


if __name__ == "__main__":
    main()
