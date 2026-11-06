"""Phase 3 - market regime detection from the correlation-to-volatility composite.

Bear (risk-off) when the correlation proxy AND the volatility-spike are both
elevated. Tuned on pre-2016; tested after. Judged by whether 'bear' days really
have worse forward returns and deeper forward drawdowns than 'bull' days.
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, pandas as pd
import phase3

OUT = "/home/user/academic_trading/results/sentiment_tuning"; SCR = os.path.dirname(__file__)
SPLIT = phase3.SPLIT


def fwd_maxdd(px, h):
    """worst drawdown over the next h days (negative)."""
    p = px.to_numpy(float); n = len(p); out = np.full(n, np.nan)
    for i in range(n):
        j = min(i + h, n - 1)
        if j > i:
            seg = p[i:j+1]; out[i] = seg.min()/p[i] - 1
    return pd.Series(out, index=px.index)


def evaluate(df, a, b, m):
    bear = (df["vspike"] > a) & (df["corr"] > b)
    sub = df[m]; bsub = bear[m]
    f = sub["fwd20"]
    return dict(a=a, b=b,
                bear_share=float(bsub.mean()),
                bull_fwd=float(f[~bsub].mean()), bear_fwd=float(f[bsub].mean()),
                sep=float(f[~bsub].mean() - f[bsub].mean()),
                bull_dd=float(sub["fdd60"][~bsub].mean()), bear_dd=float(sub["fdd60"][bsub].mean()),
                bear_fwd_negshare=float((f[bsub] < 0).mean()))


def main():
    df = phase3.market_features()
    df["fwd20"] = phase3.fwd_return(df["px"], 20)
    df["fdd60"] = fwd_maxdd(df["px"], 60)
    df = df.dropna(subset=["vspike", "corr", "fwd20", "fdd60"])
    tune = df.index < SPLIT; test = df.index >= SPLIT

    grid = list(itertools.product([0.95, 1.0, 1.05, 1.10, 1.15], [0.22, 0.26, 0.30, 0.34, 0.38]))
    best = None
    for a, b in grid:
        r = evaluate(df, a, b, tune)
        if r["bear_share"] < 0.08 or r["bear_share"] > 0.5:   # need a usable, not-too-rare bear state
            continue
        score = r["sep"]                                      # maximise bull-minus-bear forward return
        if best is None or score > best[0]:
            best = (score, a, b, r)
    _, a, b, rtune = best
    rtest = evaluate(df, a, b, test)
    rall_tune, rall_test = rtune, rtest

    print(f"MARKET REGIME (bear = vspike>{a} AND corr>{b})   tune<{SPLIT.date()}<=test")
    print(f"\n{'era':5s} {'bear%':>6s} {'bull_fwd20':>11s} {'bear_fwd20':>11s} {'separation':>11s} {'bull_dd60':>10s} {'bear_dd60':>10s} {'bear_neg%':>9s}")
    for era, r in (("tune", rtune), ("test", rtest)):
        print(f"{era:5s} {100*r['bear_share']:6.1f} {100*r['bull_fwd']:10.2f}% {100*r['bear_fwd']:10.2f}% "
              f"{100*r['sep']:10.2f}% {100*r['bull_dd']:9.1f}% {100*r['bear_dd']:9.1f}% {100*r['bear_fwd_negshare']:8.1f}%")

    # current regime
    bear_now = bool((df['vspike'].iloc[-1] > a) and (df['corr'].iloc[-1] > b))
    print(f"\nlatest date {df.index[-1].date()}: vspike={df['vspike'].iloc[-1]:.2f} corr={df['corr'].iloc[-1]:.2f} "
          f"-> regime = {'BEAR/risk-off' if bear_now else 'BULL/risk-on'}")

    # full grid table saved
    rows = [evaluate(df, a2, b2, tune) | {"era": "tune"} for a2, b2 in grid]
    rows += [evaluate(df, a2, b2, test) | {"era": "test"} for a2, b2 in grid]
    pd.DataFrame(rows).round(4).to_csv(f"{OUT}/phase3_regime_grid.csv", index=False)

    model = dict(kind="market_regime", bear_rule=dict(vspike_gt=a, corr_gt=b),
                 corr_win=60, vmaw=50, tune_before=str(SPLIT.date()),
                 tune=rtune, test=rtest, latest_regime=("bear" if bear_now else "bull"),
                 latest_date=str(df.index[-1].date()))
    json.dump(model, open(f"{OUT}/phase3_model_market.json", "w"), indent=2)
    print(f"\nwrote {OUT}/phase3_regime_grid.csv and phase3_model_market.json")


if __name__ == "__main__":
    main()
