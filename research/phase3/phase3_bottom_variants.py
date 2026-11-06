"""Compare bottom-rule variants, re-tuned on pre-2016 troughs, to fix the
April-2025 miss WITHOUT overfitting to it:
  V1  original : vspike & corr60 & dd         (the slow-correlation rule)
  V2  faster   : vspike & corr20 & dd         (20-day correlation proxy)
  V3  no-corr  : vspike & dd                  (drop correlation entirely)
Each re-tuned on train troughs; reported on test, with explicit 2020 & 2025 checks.
"""
from __future__ import annotations
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, pandas as pd, phase3

SPLIT = phase3.SPLIT; WIN = 25; PCT = 0.15


def clusters(ix):
    if len(ix) == 0: return []
    out=[]; s=p=ix[0]
    for i in ix[1:]:
        if i-p>5: out.append((s,p)); s=i
        p=i
    out.append((s,p)); return out


def score(mask, turns, era_tune, dates):
    ix = np.where(mask.to_numpy())[0]
    if era_tune is not None:
        sp = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))
        ix = ix[(ix < sp) == era_tune]
        turns = [t for t in turns if (t < sp) == era_tune]
    cl = clusters(list(ix)); hit=0; leads=[]; good=0
    for t in turns:
        near=[c for c in cl if (c[0]-WIN)<=t<=(c[1]+WIN)]
        if near: hit+=1; leads.append(min([(c[0]+c[1])//2 for c in near],key=lambda x:abs(x-t))-t)
    for c in cl:
        if any((c[0]-WIN)<=t<=(c[1]+WIN) for t in turns): good+=1
    return dict(recall=hit/max(len(turns),1), precision=good/max(len(cl),1),
                n_turns=len(turns), n_alerts=len(cl), leadlag=float(np.median(leads)) if leads else np.nan)


def main():
    d60 = phase3.market_features(corr_win=60); d20 = phase3.market_features(corr_win=20)
    df = d60.copy(); df["corr20"] = d20["corr"]; df = df.dropna(subset=["vspike","corr","corr20","dd"])
    dates = df.index
    _, troughs = phase3.zigzag(df["px"], PCT)
    tp = [int(np.where(dates==d)[0][0]) for d in troughs if d in dates]

    def tune_rule(feat, cgrid):
        best=None
        for a,c,dd in itertools.product([1.1,1.2,1.3,1.4], cgrid, [-0.10,-0.14,-0.18]):
            if feat is None:
                mask=(df["vspike"]>a)&(df["dd"]<dd)
            else:
                mask=(df["vspike"]>a)&(df[feat]>c)&(df["dd"]<dd)
            r=score(mask, tp, True, dates)
            if r["n_alerts"]==0: continue
            f=r["recall"]*r["precision"]
            if best is None or f>best[0]: best=(f,a,c,dd,mask)
        return best

    variants = {
        "V1 corr60 (original)": ("corr", [0.26,0.30,0.34]),
        "V2 corr20 (faster)":   ("corr20", [0.26,0.30,0.34,0.38]),
        "V3 no-corr":           (None, [0]),
    }
    print(f"BOTTOM-RULE VARIANTS (ZigZag {int(PCT*100)}% troughs, tune<{SPLIT.date()}<=test)\n")
    results={}
    for name,(feat,cg) in variants.items():
        f,a,c,dd,mask = tune_rule(feat, cg)
        rt=score(mask,tp,True,dates); rte=score(mask,tp,False,dates)
        results[name]=(a,c,dd,mask)
        rule = f"vspike>{a}" + (f" & {feat}>{c}" if feat else "") + f" & dd<{dd}"
        print(f"{name}:  {rule}")
        print(f"   tune: recall {rt['recall']:.0%} ({int(round(rt['recall']*rt['n_turns']))}/{rt['n_turns']}), "
              f"precision {rt['precision']:.0%} of {rt['n_alerts']}, lead/lag {rt['leadlag']:+.0f}d")
        print(f"   test: recall {rte['recall']:.0%} ({int(round(rte['recall']*rte['n_turns']))}/{rte['n_turns']}), "
              f"precision {rte['precision']:.0%} of {rte['n_alerts']}, lead/lag {rte['leadlag']:+.0f}d")
        # explicit event check: 2020-03-23 and 2025-04-08 troughs
        for ev in ["2020-03-23","2025-04-08"]:
            evd=pd.Timestamp(ev)
            if evd in dates:
                t=int(np.where(dates==evd)[0][0]); ix=np.where(mask.to_numpy())[0]
                near=[c2 for c2 in clusters(list(ix)) if (c2[0]-WIN)<=t<=(c2[1]+WIN)]
                print(f"     {ev}: {'FLAGGED' if near else 'MISSED'}")
        print()


if __name__ == "__main__":
    main()
