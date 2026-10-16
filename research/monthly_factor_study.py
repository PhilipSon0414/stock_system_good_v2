import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0,'.')
from config_v2 import *
from scipy.stats import spearmanr
pan=pd.read_pickle(PANEL_FILE)[['date','code','mcap']]
mc=pan.set_index(['code','date'])['mcap']
INDEX_FILE=CACHE_DIR/"index_ohlcv.pkl"
idxd=pickle.load(open(INDEX_FILE,'rb')); print('index keys',list(idxd))
recs=[]
for f in sorted(PX_DIR.glob('*.pkl')):
    code=f.stem
    try: px=pd.read_pickle(f)
    except: continue
    px=px.dropna(subset=['Open','High','Low','Close','Volume'])
    if len(px)<200: continue
    o,h,l,c,v=[px[k].astype(float) for k in ('Open','High','Low','Close','Volume')]
    r1=c.pct_change(); vm=v.rolling(60).mean()
    F=pd.DataFrame({
     'ret1m':c.pct_change(20),'ret3m':c.pct_change(60),'ret6m':c.pct_change(120),
     'ret5d':c.pct_change(5),
     'vol_trend':vm/vm.shift(60).replace(0,np.nan),
     'vol_ratio20':v.rolling(20).mean()/vm.replace(0,np.nan),
     'volat60':r1.rolling(60).std(),
     'dist52':c/h.rolling(240,min_periods=120).max()-1,
     'rng60':(h.rolling(60).max()-l.rolling(60).min())/c,
     'ud':(v*(c>o)).rolling(60).sum()/(v*(c<o)).rolling(60).sum().replace(0,np.nan),
     'maxup':r1.rolling(60).max(),
     'c_ma60':c/c.rolling(60).mean()-1,
     'val20':(c*v).rolling(20).mean(),
     'close':c})
    pos=np.arange(len(px)); mp=px.index.to_period('M')
    g=pd.Series(pos,index=mp).groupby(level=0).agg(['min','max','count'])
    g=g[g['count']>=15].iloc[:-1]            # 마지막(진행중) 달 제외
    g=g[g['min']>0]
    for per,(i,j,n) in g[['min','max','count']].iterrows():
        row=F.iloc[i-1].to_dict()
        if o.iloc[i]<=0 or np.isnan(row['ret6m']): continue
        row.update(code=code,month=per,ret=c.iloc[j]/o.iloc[i]-1,
                   mcap=mc.get((code,px.index[i-1]),np.nan))
        recs.append(row)
D=pd.DataFrame(recs)
D=D[(D.close>=MIN_PRICE)&(D.mcap>=MIN_MCAP)&(D.val20>=5e8)]
D['lmcap']=np.log(D.mcap); D['lval']=np.log(D.val20)
D['turn']=D.val20/D.mcap
print('rows',len(D),'months',D.month.nunique(), D.month.min(),D.month.max())
COST=0.006
bench=D.groupby('month').ret.mean()
print('유니버스 동일가중 월수익 평균 %.2f%% 중앙 %.2f%% 양(+)달 %d/%d'%(bench.mean()*100,bench.median()*100,(bench>0).sum(),len(bench)))
feats=['ret1m','ret3m','ret6m','ret5d','vol_trend','vol_ratio20','volat60','dist52','rng60','ud','maxup','c_ma60','lmcap','lval','turn']
print('\n── 단일 지표: 월별 순위상관(IC) 평균 / 상위20% 초과수익 / 하위20% 초과수익 ──')
res=[]
for f in feats:
    ics=[];tops=[];bots=[]
    for m,g in D.groupby('month'):
        gg=g[[f,'ret']].dropna()
        if len(gg)<100: continue
        ics.append(spearmanr(gg[f],gg.ret)[0])
        q=gg[f].rank(pct=True)
        tops.append(gg.ret[q>.8].mean()-gg.ret.mean()); bots.append(gg.ret[q<=.2].mean()-gg.ret.mean())
    ics=np.array(ics); res.append((f,ics.mean(),ics.mean()/ics.std()*np.sqrt(len(ics)),np.mean(tops),np.mean(bots),(np.array(tops)>0).mean()))
for f,ic,t,tp,bt,w in sorted(res,key=lambda x:-abs(x[1])):
    print(f'{f:12s} IC {ic:+.3f} (t={t:+.1f}) 상위20% {tp*100:+5.2f}%p 하위20% {bt*100:+5.2f}%p 상위가 이긴 달 {w*100:3.0f}%')
# 이벤트형 '조용한 매집'
m=(D.vol_trend>=3)&(D.ret3m.between(-.1,.1))&(D.rng60<=.5)&(D.ud>=1.3)
print('\n조용한 매집(vt≥3,3m±10%%,rng≤50%%,ud≥1.3): n=%d 월평균 %.2f%% (유니버스 %.2f%%) 중앙 %.2f%%'%(m.sum(),D.ret[m].mean()*100,D.ret.mean()*100,D.ret[m].median()*100))
# 워크포워드 모델
from sklearn.ensemble import HistGradientBoostingRegressor
D['y']=D.groupby('month').ret.rank(pct=True)
months=sorted(D.month.unique()); out=[]
for k in range(10,len(months)):
    tr=D[D.month.isin(months[:k-1])]; te=D[D.month==months[k]]   # 1개월 공백
    mdl=HistGradientBoostingRegressor(max_iter=150,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=200,random_state=0).fit(tr[feats],tr.y)
    te=te.assign(p=mdl.predict(te[feats])).sort_values('p',ascending=False)
    row={'month':months[k],'univ':te.ret.mean()}
    for n in (10,20,40): row[f'top{n}']=te.ret.head(n).mean()-COST
    row['IC']=spearmanr(te.p,te.ret)[0]; out.append(row)
W=pd.DataFrame(out)
print('\n── 워크포워드 ML(월 1회 리밸런싱, 비용 %.1f%% 차감) %d개월 ──'%(COST*100,len(W)))
print('유니버스 평균 %.2f%%/월 | IC %.3f'%(W.univ.mean()*100,W.IC.mean()))
for n in (10,20,40):
    s=W[f'top{n}']; print(f'top{n}: 월평균 {s.mean()*100:+.2f}% 중앙 {s.median()*100:+.2f}% 양(+)달 {(s>0).sum()}/{len(s)} 최악 {s.min()*100:+.1f}% 누적 {((1+s).prod()-1)*100:+.0f}% (유니버스 누적 {((1+W.univ).prod()-1)*100:+.0f}%)')
print(W.round(3).to_string())
# 시장 필터
for nm,df in idxd.items():
    cc=df['Close']; ok=(cc>cc.rolling(60).mean())
    mm={}
    for per in months:
        d0=cc.index[cc.index.to_period('M')==per]
        if len(d0)==0: continue
        prev=cc.index[cc.index<d0[0]]
        if len(prev)==0: continue
        mm[per]=bool(ok.loc[prev[-1]])
    W[f'flt_{nm}']=W.month.map(mm)
for col in [c for c in W if c.startswith('flt_')]:
    s=W.top20.where(W[col],0); print(f'{col}: 지수>60일선인 달만 투자(top20) 월평균 {s.mean()*100:+.2f}% 투자한 달 {int(W[col].sum())}/{len(W)} 누적 {((1+s).prod()-1)*100:+.0f}%')
