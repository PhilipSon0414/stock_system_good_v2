import sys, time, pandas as pd, FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'.')
from config_v2 import *
codes=[f.stem for f in PX_DIR.glob('*.pkl')]
out=str(CACHE_DIR/'px_long')+'/'
import os; os.makedirs(out,exist_ok=True)
def one(c):
    import os
    p=out+c+'.pkl'
    if os.path.exists(p): return 1
    for _ in range(3):
        try:
            df=fdr.DataReader(c,'2016-01-01')
            if df is not None and len(df)>0:
                df.to_pickle(p); return 1
        except Exception: time.sleep(1)
    return 0
t=time.time()
with ThreadPoolExecutor(12) as ex: r=list(ex.map(one,codes))
print('ok',sum(r),'/',len(codes),'sec',int(time.time()-t))
