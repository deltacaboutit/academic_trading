"""Phase 3 - tops / bottoms detector from the correlation-to-volatility composite.

BOTTOM (capitulation): volatility spike + high cross-sectional correlation + deep
drawdown -> everything selling off together on a vol surge.
TOP (complacency): low volatility + low correlation + price extended above trend
and near highs -> calm, dispersed, stretched.

Signals are causal. Ground-truth turns come from a ZigZag swing filter (used for
SCORING only). Tuned on turns before 2016, tested on turns after. Reported with
recall, precision, and median lead/lag (negative = signal leads the turn).
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/academic_trading")
os.chdir("/home/user/academic_trading")
import numpy as np, pandas as pd
import phase3

OUT = "/home/user/academic_trading/results/sentiment_tuning"
SPLIT = phase3.SPLIT
WIN = 25          # trading days: an alert within +/-WIN of a true turn counts as a hit


def alert_days(mask):
    """collapse consecutive alert days into clusters; return the representative day
    of each cluster (the day the condition is strongest handled by caller)."""
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return []
    clusters = []; start = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > 5:
            clusters.append((start, prev)); start = i
        prev = i
    clusters.append((start, prev))
    return clusters


def score(alerts_pos, true_turns_pos, n):
    """alerts_pos: list of (start,end) index clusters; true_turns_pos: list of int idx."""
    if not true_turns_pos:
        return None
    hit_turns = 0; leads = []
    for t in true_turns_pos:
        near = [c for c in alerts_pos if (c[0] - WIN) <= t <= (c[1] + WIN)]
        if near:
            hit_turns += 1
            # lead/lag of the cluster centre nearest the turn
            centres = [ (c[0]+c[1])//2 for c in near ]
            nearest = min(centres, key=lambda x: abs(x - t))
            leads.append(nearest - t)      # +ve = signal lagged the turn
    good_alerts = 0
    for c in alerts_pos:
        if any((c[0]-WIN) <= t <= (c[1]+WIN) for t in true_turns_pos):
            good_alerts += 1
    recall = hit_turns/len(true_turns_pos)
    precision = good_alerts/len(alerts_pos) if alerts_pos else np.nan
    return dict(recall=recall, precision=precision, n_turns=len(true_turns_pos),
                n_alerts=len(alerts_pos), median_leadlag=float(np.median(leads)) if leads else np.nan)


def main():
    df = phase3.market_features().dropna(subset=["vspike","corr","dd","ext"])
    pos = {d: i for i, d in enumerate(df.index)}
    for pct in (0.10, 0.15):
        peaks, troughs = phase3.zigzag(df["px"], pct)
        tp = [pos[d] for d in troughs if d in pos]; pp = [pos[d] for d in peaks if d in pos]
        tr_tune = [i for i in tp if df.index[i] < SPLIT]; tr_test = [i for i in tp if df.index[i] >= SPLIT]
        pk_tune = [i for i in pp if df.index[i] < SPLIT]; pk_test = [i for i in pp if df.index[i] >= SPLIT]

        # ---- BOTTOM detector: tune (vspike>a, corr>b, dd<d) on train troughs ----
        bestb = None
        for a, b, d in itertools.product([1.1,1.2,1.3,1.4],[0.26,0.30,0.34],[-0.08,-0.12,-0.16]):
            mask = (df["vspike"]>a)&(df["corr"]>b)&(df["dd"]<d)
            cl = alert_days(mask)
            r = score(cl, tr_tune, len(df))
            if r is None or r["n_alerts"]==0: continue
            f = r["recall"]*( r["precision"] if not np.isnan(r["precision"]) else 0)
            if bestb is None or f > bestb[0]:
                bestb = (f, a, b, d, mask, cl)
        _, ab, bb, db, bmask, bcl = bestb
        bt_tune = score(bcl, tr_tune, len(df)); bt_test = score(bcl, tr_test, len(df))

        # ---- TOP detector: tune (vspike<a, corr<b, ext>e, near highs dd>-0.05) ----
        bestt = None
        for a, b, e in itertools.product([0.92,0.98,1.04],[0.14,0.18,0.22],[0.05,0.08,0.11]):
            mask = (df["vspike"]<a)&(df["corr"]<b)&(df["ext"]>e)&(df["dd"]>-0.05)
            cl = alert_days(mask)
            r = score(cl, pk_tune, len(df))
            if r is None or r["n_alerts"]==0: continue
            f = r["recall"]*(r["precision"] if not np.isnan(r["precision"]) else 0)
            if bestt is None or f > bestt[0]:
                bestt = (f, a, b, e, mask, cl)
        _, at, bt, et, tmask, tcl = bestt
        tp_tune = score(tcl, pk_tune, len(df)); tp_test = score(tcl, pk_test, len(df))

        print("="*74)
        print(f"ZigZag {int(pct*100)}% swings:  {len(tp)} troughs, {len(pp)} peaks  (tune<{SPLIT.date()}<=test)")
        print(f"BOTTOM rule: vspike>{ab}, corr>{bb}, dd<{db}")
        for era,r in (("tune",bt_tune),("test",bt_test)):
            if r: print(f"  bottoms {era}: recall {r['recall']:.0%} ({int(round(r['recall']*r['n_turns']))}/{r['n_turns']}), "
                        f"precision {r['precision']:.0%} of {r['n_alerts']} alerts, median lead/lag {r['median_leadlag']:+.0f}d")
        print(f"TOP rule: vspike<{at}, corr<{bt}, ext>{et}, near-highs")
        for era,r in (("tune",tp_tune),("test",tp_test)):
            if r: print(f"  tops    {era}: recall {r['recall']:.0%} ({int(round(r['recall']*r['n_turns']))}/{r['n_turns']}), "
                        f"precision {r['precision']:.0%} of {r['n_alerts']} alerts, median lead/lag {r['median_leadlag']:+.0f}d")
        if pct==0.15:
            # list the major turns and whether flagged
            print("  --- major turns (15%) and detection ---")
            for i in tp:
                near=[c for c in bcl if (c[0]-WIN)<=i<=(c[1]+WIN)]
                print(f"    trough {df.index[i].date()}  {'FLAGGED' if near else 'missed '}")
            for i in pp:
                near=[c for c in tcl if (c[0]-WIN)<=i<=(c[1]+WIN)]
                print(f"    peak   {df.index[i].date()}  {'FLAGGED' if near else 'missed '}")
            model=dict(kind="tops_bottoms", zigzag_pct=0.15, window_days=WIN,
                       bottom_rule=dict(vspike_gt=ab,corr_gt=bb,dd_lt=db),
                       top_rule=dict(vspike_lt=at,corr_lt=bt,ext_gt=et,dd_gt=-0.05),
                       bottoms=dict(tune=bt_tune,test=bt_test), tops=dict(tune=tp_tune,test=tp_test))
            json.dump(model, open(f"{OUT}/phase3_model_turns.json","w"), indent=2, default=float)
    print(f"\nwrote {OUT}/phase3_model_turns.json")


if __name__ == "__main__":
    main()
