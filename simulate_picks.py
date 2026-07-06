# -*- coding: utf-8 -*-
"""
픽 수익 시뮬레이션 — 다음 단계 ② (다음날 시가 진입 기준 현실 검증)

precision@k(라벨 적중률)가 실제 매매 수익으로 이어지는지 확인한다.
워크포워드로 매월 모델을 다시 학습해 그 달의 매일 상위 k픽을 뽑고,
'다음 거래일 시가 매수' 기준으로 세 가지 청산 전략을 시뮬레이션:

  A. 1일 보유:  D+1 시가 매수 → D+1 종가 매도
  B. 5일 보유:  D+1 시가 매수 → D+5 종가 매도
  C. 목표/손절: D+1 시가 매수 → +10% 목표 / -5% 손절 / D+5 종가 청산
     (같은 날 목표·손절 모두 터치 시 손절 우선 = 보수적 가정)

갭 통계(D+1 시가 / D 종가)도 함께 — 예측 수익이 갭에 얼마나 잠식되는지.
거래비용 0.25% (수수료+제세금 왕복 근사) 차감.

사용법:
    python3 simulate_picks.py [spec] [k]     # 기본: j1 5
"""
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from config_v2 import (BASE_DIR, PX_DIR, EMBARGO_CAL_DAYS,
                       N_TEST_MONTHS)
from train_panel import load_panel, _mk_model

COST      = 0.0025      # 왕복 거래비용 근사
TARGET    = 0.10        # 전략 C 목표
STOP      = -0.05       # 전략 C 손절
HOLD_DAYS = 5

_px_cache: dict = {}


def _px(code: str):
    if code not in _px_cache:
        f = PX_DIR / f'{code}.pkl'
        _px_cache[code] = pd.read_pickle(f) if f.exists() else None
    return _px_cache[code]


def trade_outcomes(code: str, scan_date) -> dict | None:
    """scan_date 픽 하나의 세 전략 수익률 (비용 차감 전)."""
    px = _px(code)
    if px is None:
        return None
    try:
        pos = px.index.get_loc(scan_date)
    except KeyError:
        return None
    fut = px.iloc[pos + 1: pos + 1 + HOLD_DAYS]
    if len(fut) < 1:
        return None
    entry = float(fut['Open'].iloc[0])
    if entry <= 0:
        return None

    prev_close = float(px['Close'].iloc[pos])
    out = {'gap': entry / prev_close - 1,
           'ret_1d': float(fut['Close'].iloc[0]) / entry - 1}

    if len(fut) >= HOLD_DAYS:
        out['ret_5d'] = float(fut['Close'].iloc[-1]) / entry - 1
    else:
        out['ret_5d'] = float(fut['Close'].iloc[-1]) / entry - 1  # 조기 상장폐지 등

    # 전략 C: 목표/손절 (같은 날 둘 다 터치 → 손절 우선)
    ret_c = None
    for j in range(len(fut)):
        lo = float(fut['Low'].iloc[j]) / entry - 1
        hi = float(fut['High'].iloc[j]) / entry - 1
        if j == 0:
            # 진입일: 시가가 이미 목표/손절 범위 밖이면 시가 기준
            pass
        if lo <= STOP:
            ret_c = STOP
            break
        if hi >= TARGET:
            ret_c = TARGET
            break
    if ret_c is None:
        ret_c = float(fut['Close'].iloc[-1]) / entry - 1
    out['ret_ts'] = ret_c
    return out


def simulate(spec='j1', k=5, max_ret1=None):
    """max_ret1: 픽 당일 등락률 상한 — 이미 급등(상한가류)한 종목을
    '랭킹 단계'에서 제외하고 상위 k픽을 유지 (사후 필터와 달리 거래수 보존)."""
    panel, feats, label = load_panel(spec)
    months = sorted(panel['date'].dt.to_period('M').unique())
    test_months = months[-N_TEST_MONTHS:]

    hot = f', 픽당일 ret1≤{max_ret1*100:.0f}%' if max_ret1 is not None else ''
    print(f"■ 픽 수익 시뮬레이션  ({label}, 상위 {k}픽/일{hot}, "
          f"비용 {COST*100:.2f}%, 손절우선 보수 가정)\n")

    all_trades = []
    rnd_trades = []
    rng = np.random.default_rng(42)

    for m in test_months:
        te = panel[panel['date'].dt.to_period('M') == m]
        if te.empty:
            continue
        cutoff = te['date'].min() - timedelta(days=EMBARGO_CAL_DAYS)
        tr = panel[panel['date'] < cutoff]
        if len(tr) < 50_000:
            continue
        t0 = time.time()
        model = _mk_model()
        model.fit(tr[feats], tr[label])
        te = te.copy()
        te['prob'] = model.predict_proba(te[feats])[:, 1]

        pool = te if max_ret1 is None else te[te['ret1'] <= max_ret1]
        picks = (pool.sort_values('prob', ascending=False)
                   .groupby('date').head(k))
        m_tr = []
        for _, r in picks.iterrows():
            o = trade_outcomes(r['code'], r['date'])
            if o:
                o.update({'month': str(m), 'date': str(r['date'].date()),
                          'code': r['code'], 'name': r['name'],
                          'prob': float(r['prob']),
                          'ret1': float(r['ret1'])})   # 픽 당일 등락률 (필터 분석용)
                m_tr.append(o)
        all_trades += m_tr

        # 랜덤 베이스라인: 매일 무작위 k픽
        rnd = (te.assign(_r=rng.random(len(te)))
                 .sort_values('_r').groupby('date').head(k))
        for _, r in rnd.iterrows():
            o = trade_outcomes(r['code'], r['date'])
            if o:
                rnd_trades.append(o)

        df_m = pd.DataFrame(m_tr)
        if not df_m.empty:
            print(f"  {m}: {len(df_m)}건  갭 {df_m['gap'].mean()*100:+.1f}%  "
                  f"A(1일) {df_m['ret_1d'].mean()*100:+.2f}%  "
                  f"B(5일) {df_m['ret_5d'].mean()*100:+.2f}%  "
                  f"C(목표/손절) {df_m['ret_ts'].mean()*100:+.2f}%  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    df  = pd.DataFrame(all_trades)
    rdf = pd.DataFrame(rnd_trades)
    if df.empty:
        print('거래 없음'); return

    print(f"\n  ── 종합 ({len(df)}건, 비용 차감 후) ──")
    for col, nm in [('ret_1d', 'A 1일보유'), ('ret_5d', 'B 5일보유'),
                    ('ret_ts', 'C 목표/손절')]:
        r = df[col] - COST
        rr = rdf[col] - COST if not rdf.empty else pd.Series([np.nan])
        win = (r > 0).mean()
        print(f"  {nm}: 평균 {r.mean()*100:+.2f}%/건  중앙값 {r.median()*100:+.2f}%"
              f"  승률 {win*100:.0f}%  (랜덤픽 {rr.mean()*100:+.2f}%)")
    print(f"  평균 갭(시가 프리미엄): {df['gap'].mean()*100:+.2f}%"
          f"  (중앙값 {df['gap'].median()*100:+.2f}%)")

    suffix = f'_cool{int(max_ret1*100)}' if max_ret1 is not None else ''
    df.to_json(BASE_DIR / f'sim_trades_{label}_k{k}{suffix}.json',
               orient='records', indent=1, force_ascii=False)
    print(f"  상세: sim_trades_{label}_k{k}{suffix}.json")


if __name__ == '__main__':
    spec = sys.argv[1] if len(sys.argv) > 1 else 'j1'
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mr = float(sys.argv[3]) if len(sys.argv) > 3 else None
    simulate(spec, k, mr)
