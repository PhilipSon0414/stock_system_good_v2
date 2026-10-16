import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0,'.')
from config_v2 import CACHE_DIR
PX=CACHE_DIR/'px_long'
cal=pd.read_pickle(PX/'005930.pkl').index; N=len(cal); STEP=10
T=list(range(250,N-22,STEP)); print('calendar',cal[0].date(),cal[-1].date(),N,'rebal dates',len(T))
recs=[]
for f in sorted(PX.glob('*.pkl')):
    try: px=pd.read_pickle(f).reindex(cal)
    except: continue
    if px['Close'].notna().sum()<300: continue
    c=px['Close'].astype(float).ffill(); o=px['Open'].astype(float).where(px['Open']>0)
    h=px['High'].astype(float).fillna(c); v=px['Volume'].astype(float).fillna(0)
    r1=c.pct_change(); tv=c*v
    G=pd.DataFrame({'vol60':r1.rolling(60).std(),'maxup60':r1.rolling(60).max(),
      'dhigh240':c/h.rolling(240,min_periods=200).max()-1,'val20':tv.rolling(20).mean(),
      'turn_proxy':v.rolling(20).mean()/v.rolling(250,min_periods=120).mean().replace(0,np.nan),
      'close':c,'zeros20':(v==0).rolling(20).sum()})
    for H in (20,40,60): G[f'ret_{H}']=c.shift(-H)/o.shift(-1)-1
    S=G.iloc[T].copy(); S['code']=f.stem; S['date']=cal[T]; recs.append(S)
D=pd.concat(recs,ignore_index=True)
D=D[(D.close>=1000)&(D.val20>=5e8)&(D.zeros20<=2)]
for f in ('vol60','val20','turn_proxy','maxup60'): D[f+'_p']=D.groupby('date')[f].rank(pct=True)
D['year']=D.date.dt.year
print('rows',len(D),'dates',D.date.nunique(),'codes',D.code.nunique())
D.to_pickle(''+str(CACHE_DIR/'long_ds.pkl')+'')
COST=0.006
def port(H,filt,n=20,rank='val20',asc=False):
    rows=[]
    for d,g in D.groupby('date'):
        g=g.dropna(subset=[f'ret_{H}'])
        if len(g)<200: continue
        gg=g[filt(g)]
        if len(gg)<n: continue
        top=gg.sort_values(rank,ascending=asc).head(n)
        rows.append((d,top[f'ret_{H}'].mean()-COST,g[f'ret_{H}'].mean(),g[f'ret_{H}'].median()))
    r=pd.DataFrame(rows,columns=['date','p','u','um']); r['year']=r.date.dt.year; return r
def summ(r,H,tag):
    r=r.iloc[::max(1,H//10)]            # 겹치지 않게 솎음
    ex=r.p-r.u; k=H/20
    print(f'  {tag:34s} n={len(r):3d} 기간평균 {r.p.mean()*100:+5.1f}%(월환산 {r.p.mean()/k*100:+4.1f}%) 유니버스 {r.u.mean()*100:+5.1f}% 초과 {ex.mean()*100:+5.1f}%p 이긴기간 {(ex>0).mean()*100:3.0f}% 최악 {r.p.min()*100:+.0f}%')
RULE=lambda g:(g.vol60_p<=.5)&(g.dhigh240>-.25)
for H in (20,40,60):
    print(f'\n══ 규칙(변동성하위50% & 52주고점-25%내, 거래대금 큰 순 20) 보유 {H}일 ══')
    r=port(H,RULE)
    summ(r[r.date<'2024-01-01'],H,'2016~2023 (규칙 선택에 안 쓴 구간)')
    summ(r[r.date>='2024-01-01'],H,'2024~2026 (규칙 선택 구간)')
    summ(r,H,'전체')
    print('  연도별 초과수익(%p):',' '.join(f'{y}:{((g.p-g.u).mean())*100:+.1f}' for y,g in r.groupby('year')))
    print('  ── 견고성: 변형 규칙 (전체기간 초과수익 %p / 2016~23만) ──')
    for nm,fl,n in [('vol≤30%',lambda g:(g.vol60_p<=.3)&(g.dhigh240>-.25),20),('vol≤70%',lambda g:(g.vol60_p<=.7)&(g.dhigh240>-.25),20),
                    ('고점-15%내',lambda g:(g.vol60_p<=.5)&(g.dhigh240>-.15),20),('고점-35%내',lambda g:(g.vol60_p<=.5)&(g.dhigh240>-.35),20),
                    ('종목수30',RULE,30),('종목수50',RULE,50),('변동성필터만',lambda g:g.vol60_p<=.5,20),('고점필터만',lambda g:g.dhigh240>-.25,20),
                    ('+급등일無(<12%)',lambda g:(g.vol60_p<=.5)&(g.dhigh240>-.25)&(g.maxup60<.12),20)]:
        rr=port(H,fl,n); a=rr.iloc[::max(1,H//10)]; b=a[a.date<'2024-01-01']
        print(f'    {nm:14s} 전체 {((a.p-a.u).mean())*100:+5.1f}%p | 2016~23 {((b.p-b.u).mean())*100:+5.1f}%p (이긴기간 {((b.p-b.u)>0).mean()*100:.0f}%, n={len(b)})')
print('\n══ 제외 효과: 변동성 상위10%/회전율 상위10%/급등일 상위10% 종목의 평균 초과수익(%p) ══')
for H in (20,60):
    r=D.dropna(subset=[f'ret_{H}']).copy(); r['ex']=r[f'ret_{H}']-r.groupby('date')[f'ret_{H}'].transform('mean')
    for f in ('vol60_p','turn_proxy_p','maxup60_p'):
        top=r[r[f]>.9].ex.mean()*100; bot=r[r[f]<=.1].ex.mean()*100
        pre=r[(r[f]>.9)&(r.date<'2024-01-01')].ex.mean()*100
        print(f'  보유{H}일 {f}: 상위10% {top:+.1f}%p (2016~23만 {pre:+.1f}) / 하위10% {bot:+.1f}%p')
