# -*- coding: utf-8 -*-
"""
패널 모델 학습·검증 — 1단계 검증의 핵심

질문: 전시장 패널에서 학습하면 '내일~5일 내 10%+ 급등'을
      기저율 대비 몇 배(lift)로 골라낼 수 있는가?

평가: 월 단위 워크포워드 (최근 12개월). 각 검증월에 대해
      그 이전 데이터로만 학습(엠바고 10일) → 검증월의 매 거래일마다
      전 종목 확률 랭킹 → 상위 k개 중 실제 급등 비율 = precision@k.
      운용이 '매일 상위 몇 개 픽'이므로 AUC보다 이 지표가 본질.

사용법:
    python3 train_panel.py eval  [horizon]   # 워크포워드 (기본 5)
    python3 train_panel.py train [horizon]   # 전체 데이터로 최종 모델 저장
    python3 train_panel.py patterns [horizon]# 피처별 십분위 lift 표 (패턴 리포트)
"""
import sys
import json
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from config_v2 import (BASE_DIR, PANEL_FILE, MIN_PRICE, MIN_MCAP,
                       EMBARGO_CAL_DAYS, N_TEST_MONTHS, TOP_KS, SURGE_TH)

ID_COLS    = ['date', 'code', 'name', 'sector', 'close', 'mcap']
LABEL_PRE  = ('label_', 'fwd_maxret_')


def _label_col(spec) -> str:
    """'5' → label_5d (k일 내 고가 +10%)
    'j5' → label_jump_5d (단일일 종가 +10%)
    'o5' → label_o5 (D+1 시가 매수 후 5일 내 고가 +10%)
    'o1' → label_o1 (D+1 시가→종가 +5%)"""
    spec = str(spec)
    if spec.startswith('j'):
        return f'label_jump_{spec[1:]}d'
    if spec.startswith('o'):
        return f'label_{spec}'
    return f'label_{spec}d'


def load_panel(spec):
    panel = pd.read_pickle(PANEL_FILE)
    label = _label_col(spec)
    # 거래 가능 유니버스 필터 (학습·평가 공통)
    panel = panel[(panel['close'] >= MIN_PRICE) &
                  (panel['mcap'] >= MIN_MCAP) &
                  panel[label].notna()].copy()
    feats = [c for c in panel.columns
             if c not in ID_COLS and not c.startswith(LABEL_PRE)]
    panel = panel.sort_values('date').reset_index(drop=True)
    return panel, feats, label


def _mk_model():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        random_state=42)


def _prec_at_k(test: pd.DataFrame, prob: np.ndarray, label: str, ks):
    """검증월의 매 거래일 상위 k 픽 적중률 평균."""
    t = test[['date', label]].copy()
    t['prob'] = prob
    out = {}
    for k in ks:
        daily = (t.sort_values('prob', ascending=False)
                  .groupby('date').head(k)
                  .groupby('date')[label].mean())
        out[k] = float(daily.mean())
    return out


def evaluate(spec='5'):
    panel, feats, label = load_panel(spec)
    months = sorted(panel['date'].dt.to_period('M').unique())
    test_months = months[-N_TEST_MONTHS:]

    print(f"■ 워크포워드 검증  ({label}, "
          f"n={len(panel):,}, 피처 {len(feats)}개)")
    base_all = panel[label].mean()
    print(f"  전체 기저율: {base_all*100:.2f}%\n")

    rows = []
    for m in test_months:
        te_mask = panel['date'].dt.to_period('M') == m
        te = panel[te_mask]
        te_start = te['date'].min()
        cutoff = te_start - timedelta(days=EMBARGO_CAL_DAYS)
        tr = panel[panel['date'] < cutoff]
        if len(tr) < 50_000 or te[label].sum() < 5:
            continue

        model = _mk_model()
        t0 = time.time()
        model.fit(tr[feats], tr[label])
        prob = model.predict_proba(te[feats])[:, 1]

        auc = roc_auc_score(te[label], prob)
        ap  = average_precision_score(te[label], prob)
        pk  = _prec_at_k(te, prob, label, TOP_KS)
        base = te[label].mean()
        rows.append({'month': str(m), 'n_train': len(tr), 'n_test': len(te),
                     'base': base, 'auc': auc, 'ap': ap,
                     **{f'p@{k}': pk[k] for k in TOP_KS}})
        pks = '  '.join(f'p@{k} {pk[k]*100:5.1f}% (x{pk[k]/base:.1f})'
                        for k in TOP_KS)
        print(f"  {m}: 기저 {base*100:4.1f}%  AUC {auc:.3f}  {pks}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    if not rows:
        print('  검증 가능한 월 없음'); return
    df = pd.DataFrame(rows)
    print('\n  ── 평균 ──')
    print(f"  기저율 {df['base'].mean()*100:.2f}%  AUC {df['auc'].mean():.3f}")
    for k in TOP_KS:
        p = df[f'p@{k}'].mean()
        lift = (df[f'p@{k}'] / df['base']).mean()
        print(f"  precision@{k}: {p*100:.1f}%  (평균 lift x{lift:.2f})")
    df.to_json(BASE_DIR / f'walkforward_{label}.json',
               orient='records', indent=1)


def train_final(spec='5'):
    panel, feats, label = load_panel(spec)
    model = _mk_model()
    model.fit(panel[feats], panel[label])
    path = BASE_DIR / f'panel_model_{label}.pkl'
    joblib.dump({'model': model, 'features': feats, 'label': label,
                 'n': len(panel), 'trained_until': str(panel['date'].max().date()),
                 'base_rate': float(panel[label].mean())}, path)
    print(f'  저장: {path}  (n={len(panel):,})')


def patterns(spec='5'):
    """피처별 십분위 lift 표 — '패턴'을 사람이 읽을 수 있는 형태로.

    각 피처를 10분위로 잘라 분위별 급등률/기저율(lift)을 출력.
    lift가 단조 증가/감소하거나 특정 분위에서 튀면 그게 패턴이다.
    """
    panel, feats, label = load_panel(spec)
    base = panel[label].mean()
    print(f"■ 피처별 십분위 lift  ({label}, 기저 {base*100:.2f}%)\n")

    results = {}
    for f in feats:
        s = panel[f]
        if s.nunique() < 5:            # 플래그형: 값별
            g = panel.groupby(s)[label].agg(['mean', 'size'])
            g = g[g['size'] >= 200]
            results[f] = {str(k): round(v['mean']/base, 2)
                          for k, v in g.iterrows()}
            continue
        try:
            q = pd.qcut(s, 10, labels=False, duplicates='drop')
            g = panel.groupby(q)[label].mean() / base
            results[f] = {f'D{int(k)+1}': round(v, 2) for k, v in g.items()}
        except Exception:
            continue

    # 극단 분위 lift 기준 정렬 출력
    def strength(d):
        vals = list(d.values())
        return max(vals) if vals else 0
    for f in sorted(results, key=lambda x: -strength(results[x])):
        d = results[f]
        top = max(d, key=d.get)
        print(f"  {f:22s} 최고 {top}={d[top]:.2f}x   {d}")
    (BASE_DIR / f'pattern_lift_{label}.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'eval'
    spec = sys.argv[2] if len(sys.argv) > 2 else '5'
    if cmd == 'eval':
        evaluate(spec)
    elif cmd == 'train':
        train_final(spec)
    elif cmd == 'patterns':
        patterns(spec)
