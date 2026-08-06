# -*- coding: utf-8 -*-
"""
급등 역추적 패턴 연구 — 사용자 매매 스타일 반영 (1~2주 보유, +10% 익절)

미국주식에서 했던 것과 같은 접근: 2주 내 +10% 급등에 성공한 사례를 거꾸로
돌아가 (1) 급등 '전' D-7..D0 공통 패턴과 (2) 진입 '후' D+1..D+9 승자/패자
경로를 학습한다. 후자가 보유 중 매도/손절 타이밍 지표의 근거가 된다.

시간 규약 (매매 기준):
  픽 기준일 = D0 (스캔일 종가까지의 정보). 매수 = D+1 시가 = 보유일 j=0.
  retclose_j = Close[매수일+j] / 매수시가 - 1   (j = 매수 후 경과 거래일)
  hit = 매수일 포함 10거래일(j=0..9) 내 고가가 매수시가 +10% 도달 (label_o10)

산출:
  pattern_profile.json      — 모니터링 지표 룩업 테이블 (조건부 확률 등)
  reports/pattern_study.md  — 사람이 읽는 패턴 요약

사용법: python3 pattern_study.py            # 전체 재계산 (패널+모델 필요)
"""
import json

import numpy as np
import pandas as pd
import joblib

from config_v2 import BASE_DIR, PX_DIR, PANEL_FILE, MIN_PRICE, MIN_MCAP, SURGE_TH

PROFILE_FILE = BASE_DIR / 'pattern_profile.json'
OUT_MD       = BASE_DIR / 'reports' / 'pattern_study.md'
MODEL_FILE   = BASE_DIR / 'panel_model_label_o10.pkl'

# 보유 중 체크포인트 (매수 후 경과 거래일). j=9 = 만기(2주)
CHECK_JS = [1, 2, 3, 5, 7, 9]
# 경로 수익률 구간 (retclose_j 비닝)
RET_BINS = [-1.0, -0.15, -0.10, -0.07, -0.05, -0.03, -0.01,
            0.01, 0.03, 0.05, 0.08, 10.0]
TOP_PER_DAY = 20          # '픽 유사' 부분집합: 일별 모델 상위 20 (리포트와 동일)

# 후향 비교에 쓸 D0 피처 (패널에 존재하는 것만 사용)
BACK_FEATS = ['ret1', 'ret5', 'ret20', 'ret60', 'vol_ratio', 'vol5_over20',
              'quiet_streak', 'vol_recovery', 'turnover', 'dist_52w_high',
              'dist_20d_high', 'box20', 'c_over_ma20', 'ma_bull', 'rsi14',
              'close_pos', 'gap_open', 'consec_up', 'sec_surge_cnt_y']


def _events_one(px: pd.DataFrame, code: str) -> pd.DataFrame:
    """종목 하나의 (픽일 t → 매수 t+1 시가) 전향 경로 + 후향 경로."""
    o = px['Open'].astype('float64')
    h = px['High'].astype('float64')
    c = px['Close'].astype('float64')
    entry = o.shift(-1)
    out = pd.DataFrame(index=px.index)
    out['code'] = code
    out['entry'] = entry
    # 전향: 보유 j일째 종가 수익률 / j일까지 최고가 수익률
    for j in CHECK_JS:
        out[f'ret_{j}'] = c.shift(-(1 + j)) / entry - 1
        hmax = h[::-1].rolling(j + 1, min_periods=j + 1).max()[::-1].shift(-1)
        out[f'hmax_{j}'] = hmax / entry - 1
    # 후향: D0 기준 직전 j일 누적 수익률 (급등 전 가격 경로)
    for j in (1, 3, 5, 7):
        out[f'back_{j}'] = c / c.shift(j) - 1
    return out


def build_events() -> pd.DataFrame:
    files = sorted(PX_DIR.glob('*.pkl'))
    frames = []
    for i, f in enumerate(files):
        try:
            px = pd.read_pickle(f)
            frames.append(_events_one(px, f.stem))
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print(f'  경로 계산 {i+1}/{len(files)}', flush=True)
    ev = pd.concat(frames)
    ev.index.name = 'date'
    return ev.reset_index()


def _cond_table(df: pd.DataFrame) -> dict:
    """체크포인트 j별: retclose_j 구간 → P(만기 내 +10% | 아직 미도달)."""
    table = {}
    for j in CHECK_JS[:-1]:                      # j=9는 만기 자체
        sub = df[df[f'ret_{j}'].notna() & (df[f'hmax_{j}'] < SURGE_TH)]
        if len(sub) < 200:
            continue
        b = pd.cut(sub[f'ret_{j}'], RET_BINS)
        g = sub.groupby(b, observed=True).agg(
            p_hit=('hit', 'mean'), med_final=('ret_9', 'median'),
            n=('hit', 'size'))
        rows = []
        for iv, r in g.iterrows():
            if r['n'] < 30:
                continue
            rows.append({'lo': float(iv.left), 'hi': float(iv.right),
                         'p_hit': round(float(r['p_hit']), 4),
                         'med_final': round(float(r['med_final']), 4),
                         'n': int(r['n'])})
        table[str(j)] = rows
    return table


def _median_paths(df: pd.DataFrame) -> dict:
    """만기 내 +10% 성공(승자) vs 실패(패자)의 보유일별 중앙값 경로."""
    out = {}
    for name, sub in [('winner', df[df['hit'] == 1]),
                      ('loser', df[df['hit'] == 0])]:
        out[name] = {str(j): round(float(sub[f'ret_{j}'].median()), 4)
                     for j in CHECK_JS if sub[f'ret_{j}'].notna().sum() > 50}
        out[f'{name}_q25'] = {str(j): round(float(sub[f'ret_{j}'].quantile(.25)), 4)
                              for j in CHECK_JS
                              if sub[f'ret_{j}'].notna().sum() > 50}
    return out


def _backward_summary(df: pd.DataFrame, feats: list[str]) -> dict:
    """급등 전 D0 피처의 승자/패자 중앙값 비교 (역추적 공통점)."""
    win, lose = df[df['hit'] == 1], df[df['hit'] == 0]
    out = {}
    for f in feats + [f'back_{j}' for j in (1, 3, 5, 7)]:
        if f not in df.columns or df[f].notna().sum() < 500:
            continue
        w, l = float(win[f].median()), float(lose[f].median())
        out[f] = {'winner_med': round(w, 4), 'loser_med': round(l, 4)}
    return out


def run():
    print('■ 1/4 패널 로드 + o10 스코어링')
    panel = pd.read_pickle(PANEL_FILE)
    panel = panel[(panel['close'] >= MIN_PRICE) & (panel['mcap'] >= MIN_MCAP) &
                  panel['label_o10'].notna()].copy()
    bundle = joblib.load(MODEL_FILE)
    feats = bundle['features']
    panel['prob'] = bundle['model'].predict_proba(panel[feats])[:, 1]
    keep = ['date', 'code', 'label_o10', 'prob'] + \
        [f for f in BACK_FEATS if f in panel.columns]
    panel = panel[keep]

    print('■ 2/4 진입 후 경로 계산 (전 종목)')
    ev = build_events()
    df = panel.merge(ev, on=['date', 'code'], how='inner')
    df = df.rename(columns={'label_o10': 'hit'})
    df = df[df['ret_9'].notna()]
    print(f'  이벤트 {len(df):,}건 | 기저 hit율 {df["hit"].mean()*100:.1f}%')

    # '픽 유사' 부분집합: 일별 모델 상위 TOP_PER_DAY — 실제 픽 분포와 유사
    # (주의: 전체 학습 모델의 in-sample 점수라 확률 자체는 낙관 편향.
    #  여기선 '어떤 종목이 픽으로 뽑히는가'의 분포 근사로만 사용)
    top = (df.sort_values('prob', ascending=False)
             .groupby('date').head(TOP_PER_DAY))
    print(f'  픽 유사 부분집합 {len(top):,}건 | hit율 {top["hit"].mean()*100:.1f}%')

    print('■ 3/4 패턴 추출')
    profile = {
        'meta': {
            'label': 'label_o10 (D+1 시가 매수 → 10거래일 내 고가 +10%)',
            'trained_until': bundle.get('trained_until'),
            'check_js': CHECK_JS,
            'base_hit_all': round(float(df['hit'].mean()), 4),
            'base_hit_picklike': round(float(top['hit'].mean()), 4),
            'n_all': len(df), 'n_picklike': len(top),
        },
        'cond_all': _cond_table(df),
        'cond_picklike': _cond_table(top),
        'paths_all': _median_paths(df),
        'paths_picklike': _median_paths(top),
        'backward_picklike': _backward_summary(top, BACK_FEATS),
    }
    PROFILE_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=1),
                            encoding='utf-8')
    print(f'  저장: {PROFILE_FILE}')

    print('■ 4/4 리포트')
    _write_md(profile)


def _fmt_pct(x):
    return f'{x*100:+.1f}%'


def _write_md(pf: dict):
    m = pf['meta']
    L = [f'# 급등 역추적 패턴 연구 — {m["label"]}', '',
         f'전 시장 이벤트 {m["n_all"]:,}건 (기저 성공률 {m["base_hit_all"]*100:.1f}%) | '
         f'픽 유사 상위 {m["n_picklike"]:,}건 (성공률 {m["base_hit_picklike"]*100:.1f}%)',
         '', '## 1. 급등 "전" 공통점 (D-7 → D0, 픽 유사 집합의 승자 vs 패자 중앙값)', '',
         '| 피처 | 승자 | 패자 |', '|---|---|---|']
    for f, v in pf['backward_picklike'].items():
        L.append(f"| {f} | {v['winner_med']} | {v['loser_med']} |")

    L += ['', '## 2. 진입 "후" 경로 (보유 j일째 종가 수익률 중앙값)', '',
          '| 경과일 | 승자 중앙값 | 승자 하위25% | 패자 중앙값 |', '|---|---|---|---|']
    p = pf['paths_picklike']
    for j in map(str, m['check_js']):
        if j in p.get('winner', {}):
            L.append(f"| j={j} | {_fmt_pct(p['winner'][j])} "
                     f"| {_fmt_pct(p['winner_q25'].get(j, float('nan')))} "
                     f"| {_fmt_pct(p['loser'].get(j, float('nan')))} |")

    L += ['', '## 3. 조건부 성공 확률 — 모니터링 지표의 근거', '',
          '보유 j일째 아직 +10% 미도달일 때, 현재 수익률 구간별 '
          '"만기(j=9)까지 +10% 도달" 확률 (픽 유사 집합):', '']
    for j, rows in pf['cond_picklike'].items():
        L.append(f'**j={j}일째** (미도달 상태)')
        L.append('')
        L.append('| 현재 수익률 | P(성공) | 만기 중앙값 | n |')
        L.append('|---|---|---|---|')
        for r in rows:
            L.append(f"| {_fmt_pct(r['lo'])} ~ {_fmt_pct(r['hi'])} "
                     f"| {r['p_hit']*100:.0f}% | {_fmt_pct(r['med_final'])} "
                     f"| {r['n']:,} |")
        L.append('')

    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print(f'  저장: {OUT_MD}')


if __name__ == '__main__':
    run()
