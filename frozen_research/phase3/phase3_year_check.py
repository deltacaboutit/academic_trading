import sys, os; sys.path.insert(0,'/home/user/academic_trading/research/phase3'); sys.path.insert(0,'/home/user/academic_trading')
os.chdir('/home/user/academic_trading')
import numpy as np, pandas as pd, driver, phase3
YEAR=int(sys.argv[1]); WIN=25; PCT=0.20
spy=pd.read_parquet('price_data/SPY.parquet')['Close']; spy.index=pd.to_datetime(spy.index).normalize()
vix=pd.read_parquet('price_data/_VIX.parquet')['Close']; vix.index=pd.to_datetime(vix.index).normalize()
panel=driver.build_panel_series(spy); dates=pd.DatetimeIndex(panel.dates); C=panel.C; J=C.shape[1]
vix=vix.reindex(dates)
# ---- market ----
df=phase3.market_features().dropna(subset=['vspike','corr','dd','ext'])
y=df.loc[str(YEAR)]
print(f"=== MARKET {YEAR} (through {y.index.max().date()}) ===")
print("SPY start %.0f end %.0f max %.0f min %.0f | deepest DD %.1f%% on %s | max vspike %.2f (VIX %.1f) | max corr %.2f"%(
 y['px'].iloc[0],y['px'].iloc[-1],y['px'].max(),y['px'].min(),100*y['dd'].min(),y['dd'].idxmin().date(),
 y['vspike'].max(),y.loc[y['vspike'].idxmax(),'vix'],y['corr'].max()))
BOT=(df['vspike']>1.3)&(df['corr20']>0.26)&(df['dd']<-0.18)
for pct in (0.10,0.15):
    pk,tr=phase3.zigzag(df['px'],pct)
    t=[d.date().isoformat() for d in tr if d.year==YEAR]; p=[d.date().isoformat() for d in pk if d.year==YEAR]
    print(f"  ZigZag {int(pct*100)}% confirmed in {YEAR}: troughs={t} peaks={p}")
fy=df.loc[str(YEAR)][BOT.loc[str(YEAR)]]
print(f"  market BOTTOM-FLAG (15% rule) days in {YEAR}: {len(fy)}")
if len(fy): print(fy[['vix','vspike','corr','dd']].round(2).to_string().replace('\n','\n   '))
# ---- stocks ----
lr=pd.DataFrame(np.vstack([np.full((1,J),np.nan),np.log(C[1:]/C[:-1])]),index=dates)
rv=lr.rolling(20).std()*np.sqrt(252); vspike=(rv/rv.rolling(50).mean()).to_numpy()
Cdf=pd.DataFrame(C,index=dates); dd=(Cdf/Cdf.cummax()-1).to_numpy(); mkt=(vix/vix.rolling(50).mean()).to_numpy()
flag=(vspike>1.4)&(dd<-0.20)&(mkt[:,None]>1.0)
isY=np.array([d.year==YEAR for d in dates])
def clusters(ix):
    if len(ix)==0: return []
    out=[]; s=p=ix[0]
    for i in ix[1:]:
        if i-p>5: out.append((s,p)); s=i
        p=i
    out.append((s,p)); return out
hit=nt=good=ncl=0; nflag=0; leads=[]; tick=panel.tickers
for j in range(J):
    s=Cdf.iloc[:,j].dropna()
    if len(s)<300: continue
    _,tr=phase3.zigzag(s,PCT)
    trpos=[int(np.where(dates==d)[0][0]) for d in tr if d in dates and d.year==YEAR]
    ix=list(np.where(flag[:,j]&isY)[0]); cl=clusters(ix); nt+=len(trpos); ncl+=len(cl); nflag+= (1 if cl else 0)
    for t in trpos:
        near=[c for c in cl if (c[0]-WIN)<=t<=(c[1]+WIN)]
        if near: hit+=1; leads.append(min([(c[0]+c[1])//2 for c in near],key=lambda x:abs(x-t))-t)
    for c in cl:
        if any((c[0]-WIN)<=t<=(c[1]+WIN) for t in trpos): good+=1
print(f"\n=== STOCKS {YEAR} ===")
print(f"stocks with confirmed 20% low: {nt} | stocks firing a flag: {nflag} (clusters {ncl})")
print(f"  recall {hit}/{nt}={100*hit/max(nt,1):.0f}% | precision {good}/{ncl}={100*good/max(ncl,1):.0f}% | median lead/lag {np.median(leads):+.0f}d" if leads else f"  recall 0/{nt}")
