# -*- coding: utf-8 -*-
"""
보유 포지션 모니터링 — 학습된 급등 경로 대비 현재 위치 → 매도/손절 신호

pattern_study.py가 학습한 승자/패자 경로와 조건부 성공 확률표를 근거로,
보유 중인 종목이 "학습된 길"을 잘 가고 있는지 매일 점검한다.

포지션 등록 (positions.json — 실제 매수를 기록):
    python3 monitor_positions.py add <코드> <매수일 YYYY-MM-DD> [매수가]
      매수가 생략 시 해당일 시가 사용
    python3 monitor_positions.py close <코드>       # 매도 완료 처리
    python3 monitor_positions.py                    # 점검 리포트

positions.json이 비어 있으면 최근 매매픽(✅)을 '가상 포지션'으로 자동
추적한다 — 실제 매수 전에도 신호 품질을 관찰할 수 있게.

신호 체계 (매수일 = j0, 만기 = j9 ≈ 2주):
  🎯 익절   — 보유 중 고가가 +10% 도달 (스타일 규칙: 매도)
  ⌛ 만기   — j9 도달, +10% 미도달 (스타일 규칙: 청산, 홀딩 연장 근거 없음)
  ✂️ 손절   — 조건부 성공 확률 < 15% (패자 경로 확정권)
  ⚠️ 주의   — 확률 15~30% 또는 패자 중앙값 하회
  ⏳ 보유   — 그 외 (승자 경로 대비 위치 병기)
"""
import sys
import json
from datetime import datetime

import pandas as pd

from config_v2 import BASE_DIR, PX_DIR, SURGE_TH

POSITIONS_FILE = BASE_DIR / 'positions.json'
PROFILE_FILE   = BASE_DIR / 'pattern_profile.json'
PICKS_FILE     = BASE_DIR / 'picks_history.json'
REPORT_DIR     = BASE_DIR / 'reports'
MAX_J          = 9                 # 만기: 매수일 포함 10거래일
P_STOP, P_WARN = 0.15, 0.30


def _load_positions() -> list[dict]:
    if POSITIONS_FILE.exists():
        pos = json.loads(POSITIONS_FILE.read_text(encoding='utf-8'))
        if pos:
            return pos
    # 실포지션이 없으면 최근 매매픽을 가상 추적
    if not PICKS_FILE.exists():
        return []
    hist = json.loads(PICKS_FILE.read_text(encoding='utf-8'))
    out = []
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=25)
    for day in hist:
        if pd.Timestamp(day['date']) < cutoff:
            continue
        for p in day['picks']:
            if p.get('tradable'):
                out.append({'code': p['code'], 'name': p['name'],
                            'entry_date': None,       # 픽 다음 거래일
                            'pick_date': day['date'],
                            'entry_price': None, 'virtual': True})
    return out


def _save_positions(pos: list[dict]):
    POSITIONS_FILE.write_text(json.dumps(pos, ensure_ascii=False, indent=1),
                              encoding='utf-8')


def _px(code: str) -> pd.DataFrame | None:
    """일봉: 캐시 우선, 가능하면 오늘자 갱신."""
    f = PX_DIR / f'{code}.pkl'
    px = pd.read_pickle(f) if f.exists() else None
    try:
        import FinanceDataReader as fdr
        fresh = fdr.DataReader(code, '2026-01-01')
        if px is not None:
            px = pd.concat([px[px.index < fresh.index[0]], fresh])
        else:
            px = fresh
    except Exception:
        pass
    return px


def _lookup(table: dict, j: int, ret: float):
    """조건부 확률표에서 (경과일 j, 현재 수익률) → (P, 만기중앙값). 근사 j 사용."""
    if not table:
        return None, None
    js = sorted(int(k) for k in table.keys())
    jn = min(js, key=lambda x: abs(x - j))
    for r in table[str(jn)]:
        if r['lo'] <= ret < r['hi']:
            return r['p_hit'], r['med_final']
    return None, None


def _evaluate(p: dict, profile: dict) -> dict | None:
    px = _px(p['code'])
    if px is None or px.empty:
        return None
    # 매수일 결정: entry_date 명시 → 그 날, 가상 픽 → 픽 다음 거래일
    if p.get('entry_date'):
        after = px[px.index >= pd.Timestamp(p['entry_date'])]
    else:
        after = px[px.index > pd.Timestamp(p['pick_date'])]
    if after.empty:
        return {**p, 'status': '진입 대기', 'j': -1}
    entry_day = after.index[0]
    entry = float(p['entry_price']) if p.get('entry_price') \
        else float(after['Open'].iloc[0])
    held = px.loc[entry_day:]
    j = len(held) - 1                       # 매수일 = j0
    cur = float(held['Close'].iloc[-1])
    ret = cur / entry - 1
    hmax = float(held['High'].max()) / entry - 1
    # 익절 판정은 스타일 만기(j9) 이내 고가만 기준
    hit = float(held['High'].iloc[:MAX_J + 1].max()) / entry - 1 >= SURGE_TH

    table = profile.get('cond_picklike') or profile.get('cond_all')
    paths = profile.get('paths_picklike') or profile.get('paths_all') or {}
    p_hit, med_final = _lookup(table, min(j, MAX_J - 1), ret)
    wj = paths.get('winner', {}).get(str(min(j, MAX_J)))
    lj = paths.get('loser', {}).get(str(min(j, MAX_J)))

    if hit:
        sig, act = '🎯 익절', f'고가 {hmax*100:+.1f}% 도달 — 매도 (스타일 규칙)'
    elif j >= MAX_J:
        sig, act = '⌛ 만기', '2주 경과 — 청산 권고 (홀딩 연장 근거 없음)'
    elif p_hit is not None and p_hit < P_STOP:
        sig, act = '✂️ 손절', f'성공확률 {p_hit*100:.0f}% — 손절 권고'
    elif (p_hit is not None and p_hit < P_WARN) or \
         (lj is not None and ret < lj):
        sig = '⚠️ 주의'
        act = (f'성공확률 {p_hit*100:.0f}%' if p_hit is not None else '') + \
              (f' | 패자 경로({lj*100:+.1f}%) 하회' if lj is not None and ret < lj
               else '') + ' — 내일 회복 없으면 손절 검토'
    else:
        pos_txt = ''
        if wj is not None:
            pos_txt = ' | 승자 경로 상회' if ret >= wj else \
                      f' | 승자 중앙값({wj*100:+.1f}%) 하회'
        act = (f'성공확률 {p_hit*100:.0f}%' if p_hit is not None else
               '경로 정상') + pos_txt
        sig = '⏳ 보유'

    return {**p, 'status': '보유중', 'j': j, 'entry': entry,
            'entry_day': str(entry_day.date()), 'cur': cur, 'ret': ret,
            'hmax': hmax, 'p_hit': p_hit, 'med_final': med_final,
            'winner_j': wj, 'signal': sig, 'action': act}


def run():
    profile = json.loads(PROFILE_FILE.read_text(encoding='utf-8')) \
        if PROFILE_FILE.exists() else {}
    if not profile:
        print('⚠ pattern_profile.json 없음 — pattern_study.py 먼저 실행')
    positions = _load_positions()
    if not positions:
        print('추적할 포지션 없음 (positions.json 비어있고 최근 매매픽 없음)')
        return

    virtual = all(p.get('virtual') for p in positions)
    rows = [r for p in positions if (r := _evaluate(p, profile))]
    today = datetime.now().strftime('%Y-%m-%d')

    L = [f'# 포지션 모니터링 — {today}', '']
    if virtual:
        L += ['(실포지션 미등록 — 최근 매매픽을 가상 추적 중. 실제 매수 시 '
              '`python3 monitor_positions.py add <코드> <매수일> [매수가]`)', '']
    L += ['| 종목 | 코드 | 매수일 | 경과 | 매수가 | 현재가 | 수익률 | 최고 | '
          '성공확률 | 신호 |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['j'] < 0:
            L.append(f"| {r.get('name','')} | {r['code']} | (내일 시가) | — | — "
                     f"| — | — | — | — | 진입 대기 |")
            continue
        ph = f"{r['p_hit']*100:.0f}%" if r['p_hit'] is not None else '—'
        L.append(f"| {r.get('name','')} | {r['code']} | {r['entry_day']} "
                 f"| j{r['j']} | {r['entry']:,.0f} | {r['cur']:,.0f} "
                 f"| {r['ret']*100:+.1f}% | {r['hmax']*100:+.1f}% "
                 f"| {ph} | {r['signal']} |")
    L.append('')
    for r in rows:
        if r['j'] >= 0:
            L.append(f"- **{r.get('name','')}** ({r['code']}): {r['action']}")

    report = '\n'.join(L)
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f'positions_{today}.md'
    out.write_text(report, encoding='utf-8')
    print(report)
    print(f'\n저장: {out}')


def add(code: str, entry_date: str, price: str | None):
    pos = json.loads(POSITIONS_FILE.read_text(encoding='utf-8')) \
        if POSITIONS_FILE.exists() else []
    pos = [p for p in pos if p['code'] != code]
    name = ''
    try:
        panel_names = pd.read_pickle(BASE_DIR / 'cache' / 'krx_listing.pkl')
        panel_names['Code'] = panel_names['Code'].astype(str).str.zfill(6)
        m = panel_names[panel_names['Code'] == code]
        name = m['Name'].iloc[0] if len(m) else ''
    except Exception:
        pass
    pos.append({'code': code, 'name': name, 'entry_date': entry_date,
                'entry_price': float(price) if price else None,
                'virtual': False})
    _save_positions(pos)
    print(f'등록: {name}({code}) 매수일 {entry_date} '
          f'매수가 {price or "(시가 자동)"}')


def close(code: str):
    if not POSITIONS_FILE.exists():
        return
    pos = json.loads(POSITIONS_FILE.read_text(encoding='utf-8'))
    pos = [p for p in pos if p['code'] != code]
    _save_positions(pos)
    print(f'종료 처리: {code}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == 'add':
        add(args[1], args[2], args[3] if len(args) > 3 else None)
    elif args and args[0] == 'close':
        close(args[1])
    else:
        run()
