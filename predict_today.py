# -*- coding: utf-8 -*-
"""
다음날 급등 후보 리포트 — 매일 밤 전 종목 스코어링 (다음 단계 ①)

흐름:
  1. 종목별 일봉 캐시 갱신 (뒤처진 종목만 재수집, ~수 분)
  2. 패널 재생성 (피처 + 횡단면 순위)
  3. 최신 거래일 전 종목에 panel_model(j1, 5d) 확률 부여
  4. 상위 후보 리포트 저장 + picks_history.json 누적 (사후 검증용)

사용법:
    python3 predict_today.py                # 갱신 + 리포트
    python3 predict_today.py --no-refresh   # 캐시·패널 갱신 생략, panel.pkl 재사용
    python3 predict_today.py --top 30       # 후보 수 조정

v1(stock_system)과의 결합: 이 리포트가 '후보 생성기' 역할.
상위 후보를 v1의 세력/수급/티어 분석에 넣어 정밀 검증하는 2단 구조.
"""
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config_v2 import BASE_DIR, PANEL_FILE, MIN_PRICE, MIN_MCAP
import build_panel as bp

REPORT_DIR   = BASE_DIR / 'reports'
PICKS_FILE   = BASE_DIR / 'picks_history.json'
REPORT_DIR.mkdir(exist_ok=True)

MODELS = {'j1': 'panel_model_label_jump_1d.pkl',
          '5d': 'panel_model_label_5d.pkl',
          'o5': 'panel_model_label_o5.pkl',    # D+1 시가 매수 → 5일 내 +10%
          'o10': 'panel_model_label_o10.pkl'}  # D+1 시가 매수 → 2주 내 +10%
                                               # (사용자 스타일: 1~2주 보유 익절)

# 시뮬레이션 검증(12개월)으로 확정한 매매 규칙:
#   전체 top5 랭킹 중 '픽 당일 +15% 미만'(비상한가 제외) 그리고
#   '5일 수익률 > -30%'(falling knife 제외)만 D+1 시가 매수, 5일 보유.
#   칼날 제외 후 o5: +1.27%/건·승률45% / 완만한 눌림(-30~-10%)이 +2.78% 최적.
#   폭락주(-30%↓) 반등 베팅은 백테스트 -4.21%/건 — v1도 같은 결함을 겪고 수정
#   (2026-06 '폭락주 외삽 결함'). 생존편향(상폐주 부재)까지 감안하면 더 나쁨.
TRADABLE_RET1 = 0.15
KNIFE_RET5    = -0.30
PROFILE_FILE  = BASE_DIR / 'pattern_profile.json'
MAX_GAP       = 0.08     # 시가 갭 상한 — 이 이상 뜨면 진입 보류 (갭이 목표 잠식)


def _is_tradable(r) -> bool:
    return (r['ret1'] <= TRADABLE_RET1) and (r['ret5'] > KNIFE_RET5)


def _load_profile() -> dict:
    try:
        return json.loads(PROFILE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _stop_lines(profile: dict) -> dict[int, float]:
    """체크포인트 j별 손절선: 성공확률 15% 미만으로 떨어지는 수익률 상단."""
    table = profile.get('cond_picklike') or profile.get('cond_all') or {}
    out = {}
    for j, rows in table.items():
        his = [r['hi'] for r in rows if r['p_hit'] < 0.15]
        if his:
            out[int(j)] = max(his)
    return out


def _coaching(r, profile: dict, tradable: bool) -> list[str]:
    """픽 하나에 대한 매수/매도 타이밍 코칭 (pattern_profile 기반)."""
    paths = profile.get('paths_picklike') or {}
    win = paths.get('winner', {})
    stops = _stop_lines(profile)
    L = [f"**{r['name']} ({r['code']}) — 종가 {r['close']:,.0f}원, "
         f"2주 내 +10% 확률 {r.get('prob_o10', float('nan'))*100:.0f}%**"]

    # 진입 코칭
    if tradable:
        L.append(f"- 진입: 내일 **시가 매수**. 단 시가 갭 **+{MAX_GAP*100:.0f}% "
                 f"이상**이면 진입 보류 — 갭이 목표수익(+10%)을 잠식")
    else:
        why = []
        if r['ret1'] > TRADABLE_RET1:
            why.append(f"당일 {r['ret1']*100:+.1f}% 과열(상한가류)")
        if r['ret5'] <= KNIFE_RET5:
            why.append(f"5일 {r['ret5']*100:+.1f}% 폭락주")
        L.append(f"- 관찰용 (매매 제외: {', '.join(why) or 'top5 밖'}) — "
                 f"눌림(-3~-7%) 시 재평가")
    # 개별 특성 코멘트
    if r['ret1'] > 0.20:
        L.append(f"- ⚠ 당일 {r['ret1']*100:+.0f}% 급등 직후 — 시초가 갭 리스크 큼")
    if -0.30 < r['ret5'] <= -0.10:
        L.append(f"- 완만한 눌림({r['ret5']*100:+.0f}%) 구간 — 백테스트상 "
                 f"최적 유형(+2.78%/건)")
    if r['vol_ratio'] >= 10:
        L.append(f"- 거래량 {r['vol_ratio']:.0f}x 폭증 — 변동성 극대, 분할 진입 권장")

    # 매도/손절 계획
    L.append("- 익절: 매수가 **+10% 도달 즉시 매도** (지정가 걸어두기). "
             "10거래일(2주) 내 미도달 시 만기 청산")
    if win:
        path_txt = ' / '.join(f"j{j} {win[str(j)]*100:+.1f}%"
                              for j in (1, 3, 5) if str(j) in win)
        L.append(f"- 순항 기준(승자 중앙값): {path_txt} — 이 위면 홀딩")
    if stops:
        stop_txt = ' / '.join(f"j{j} {stops[j]*100:.0f}%"
                              for j in sorted(stops))
        L.append(f"- 손절선(성공확률 15%↓): 종가가 {stop_txt} 이탈 시 익일 손절")
    L.append("- 매일 신호 확인: 포지션 모니터링 리포트(🎯익절/⌛만기/✂️손절/"
             "⚠️주의/⏳보유)가 자동 판정")
    return L


def _load_models():
    out = {}
    for key, fn in MODELS.items():
        p = BASE_DIR / fn
        if p.exists():
            out[key] = joblib.load(p)
    if not out:
        raise SystemExit('모델 없음 — 먼저 train_panel.py train j1 / train 5 실행')
    return out


def _v1_check(codes: list[str]) -> dict:
    """v1 세력/티어 정밀 검증 (별도 프로세스, 실패해도 리포트는 계속)."""
    if not codes:
        return {}
    try:
        r = subprocess.run(
            [sys.executable, '-W', 'ignore',
             str(BASE_DIR / 'v1_bridge.py'), *codes],
            capture_output=True, text=True, timeout=600)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        print(f'  ⚠ v1 브리지 실패 (스킵): {e}')
        return {}


def run(refresh: bool = True, top_n: int = 20):
    if refresh:
        print('■ 1/3 데이터 갱신')
        uni = bp.fetch_listing()
        bp.fetch_prices(uni, refresh=True)

    if not refresh and PANEL_FILE.exists():
        # run_nightly.sh가 학습 직전에 패널을 이미 재생성 — 그대로 재사용
        print('■ 2/3 패널 로드 (기존 panel.pkl)')
        panel = pd.read_pickle(PANEL_FILE)
    else:
        print('■ 2/3 패널 재생성')
        panel = bp.build_panel()

    print('■ 3/3 스코어링')
    latest = panel['date'].max()
    day = panel[(panel['date'] == latest) &
                (panel['close'] >= MIN_PRICE) &
                (panel['mcap'] >= MIN_MCAP)].copy()

    models = _load_models()
    for key, bundle in models.items():
        feats = bundle['features']
        day[f'prob_{key}'] = bundle['model'].predict_proba(day[feats])[:, 1]

    # 정렬: 사용자 스타일(1~2주 내 +10%) = o10 우선, 없으면 o5
    sort_key = 'prob_o10' if 'prob_o10' in day.columns else 'prob_o5'
    day = day.sort_values(sort_key, ascending=False)
    picks = day.head(top_n)
    # 매매 대상: 전체 top5 안에 들면서 당일 아직 +15% 미만 (시뮬레이션 검증 규칙)
    top5_idx = set(day.head(5).index)

    # v1 세력/티어 정밀 검증 — 상위 10개만 (개당 수 초)
    print('  v1 정밀 검증 중 (상위 10)...')
    v1 = _v1_check([r['code'] for _, r in picks.head(10).iterrows()])

    # ── 리포트 ──
    date_str = str(latest.date())
    sort_txt = ('2주(10거래일) 내 +10% 확률(o10)' if sort_key == 'prob_o10'
                else '5일 내 +10% 확률(o5)')
    lines = [
        f'# 급등 후보 리포트 — 기준일 {date_str} (다음 거래일 예측)',
        '',
        f'유니버스 {len(day):,}종목 | 정렬: D+1 시가 매수 후 {sort_txt} 순',
        '',
        '**매매 규칙**: 아래 top5 중 `매매` 표시만 다음날 **시가 매수 →',
        '1~2주 보유, +10% 도달 시 익절** (보유 중 판단은 포지션 모니터링 참고).',
        '표시 조건: 당일 +15% 미만(상한가류 제외 — 갭이 수익 잠식) & 5일 -30%',
        '초과(폭락주 제외 — 12개월 시뮬레이션 검증, 백테스트 -4.2%/건).',
        '',
        '| # | 매매 | 종목 | 코드 | 종가 | o10확률 | o5확률 | j1확률 | 당일% | 5일% | 거래량비 | 업종어제급등 | v1 세력/티어 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for rank, (idx, r) in enumerate(picks.iterrows(), 1):
        tradable = '✅' if (idx in top5_idx and _is_tradable(r)) else ''
        v = v1.get(r['code'])
        v1_txt = (f"세력{v['seoryeok']}/T-{v['tier']}" if v
                  else ('게이트미달' if r['code'] in v1 else '—'))
        lines.append(
            f"| {rank} | {tradable} | {r['name']} | {r['code']} | {r['close']:,.0f} "
            f"| {r.get('prob_o10', np.nan)*100:.1f}% "
            f"| {r['prob_o5']*100:.1f}% | {r.get('prob_j1', np.nan)*100:.1f}% "
            f"| {r['ret1']*100:+.1f}% | {r['ret5']*100:+.1f}% "
            f"| {r['vol_ratio']:.1f}x | {r.get('sec_surge_cnt_y', 0):.0f} "
            f"| {v1_txt} |")

    # v1 진입 타이밍 코멘트 (매매픽만)
    timing = [(r['name'], v1.get(r['code']))
              for idx, r in picks.head(5).iterrows()
              if idx in top5_idx and _is_tradable(r)]
    timing = [(n, v) for n, v in timing if v and v.get('entry_timing')]
    if timing:
        lines += ['', '**v1 진입 타이밍 (매매픽)**:']
        lines += [f'- {n}: {v["entry_timing"]}' for n, v in timing]

    # ── 매수/매도 타이밍 코칭 (패턴 프로필 기반) ──
    profile = _load_profile()
    coach_map = {}                     # code → 코칭 텍스트 (history 저장용)
    if profile:
        lines += ['', '## 매매 코칭 (급등 역추적 패턴 기반)', '',
                  '보유 경과일 j는 매수일=j0 기준. 상세 근거: '
                  'reports/pattern_study.md', '']
        for idx, r in picks.head(5).iterrows():
            tradable = bool(idx in top5_idx and _is_tradable(r))
            block = _coaching(r, profile, tradable)
            coach_map[r['code']] = block
            lines += block + ['']

    report = '\n'.join(lines)
    out_md = REPORT_DIR / f'picks_{date_str}.md'
    out_md.write_text(report, encoding='utf-8')
    print(report)
    print(f'\n저장: {out_md}')

    # ── 사후 검증용 누적 기록 ──
    hist = []
    if PICKS_FILE.exists():
        hist = json.loads(PICKS_FILE.read_text(encoding='utf-8'))
    hist = [h for h in hist if h['date'] != date_str]     # 같은 날 재실행 대체
    hist.append({
        'date': date_str,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'picks': [{
            'rank': i + 1, 'code': r['code'], 'name': r['name'],
            'close': float(r['close']),
            'prob_o10': round(float(r.get('prob_o10', np.nan)), 4),
            'prob_o5': round(float(r.get('prob_o5', np.nan)), 4),
            'prob_j1': round(float(r.get('prob_j1', np.nan)), 4),
            'prob_5d': round(float(r.get('prob_5d', np.nan)), 4),
            'ret1': round(float(r['ret1']), 4),
            'vol_ratio': round(float(r['vol_ratio']), 2),
            'tradable': bool(idx in top5_idx and _is_tradable(r)),
            'coaching': coach_map.get(r['code']),
            'v1': v1.get(r['code']),
        } for i, (idx, r) in enumerate(picks.iterrows())],
    })
    PICKS_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=1),
                          encoding='utf-8')
    print(f'누적 기록: {PICKS_FILE} ({len(hist)}일)')


if __name__ == '__main__':
    args = sys.argv[1:]
    top = int(args[args.index('--top') + 1]) if '--top' in args else 20
    run(refresh='--no-refresh' not in args, top_n=top)
