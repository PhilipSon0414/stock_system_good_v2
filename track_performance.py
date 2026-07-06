# -*- coding: utf-8 -*-
"""
픽 성과 추적 — picks_history.json 의 과거 픽을 실제 시세로 사후 검증

매일 밤 predict_today 뒤에 실행. 각 픽에 대해:
  - 매매픽(✅): D+1 시가 진입 → 5거래일 보유 수익률 (규칙 그대로)
  - 전체픽: 5일 내 고가 +10% 터치 여부 (예측 적중률)
윈도우 미성숙 픽은 '진행중'으로 표시.

사용법: python3 track_performance.py
출력:  reports/performance.md + 콘솔 요약
"""
import json

import pandas as pd

from config_v2 import BASE_DIR, PX_DIR

PICKS_FILE = BASE_DIR / 'picks_history.json'
OUT_MD     = BASE_DIR / 'reports' / 'performance.md'
COST       = 0.0025
HOLD       = 5

_px_cache: dict = {}


def _px(code):
    if code not in _px_cache:
        f = PX_DIR / f'{code}.pkl'
        _px_cache[code] = pd.read_pickle(f) if f.exists() else None
    return _px_cache[code]


def outcome(code: str, pick_date: str) -> dict | None:
    """픽 이후 실현 성과. 데이터 없으면 None."""
    px = _px(code)
    if px is None:
        return None
    try:
        pos = px.index.get_loc(pd.Timestamp(pick_date))
    except KeyError:
        return None
    fut = px.iloc[pos + 1: pos + 1 + HOLD]
    if len(fut) < 1:
        return {'status': '진행중'}          # D+1 미도래
    entry = float(fut['Open'].iloc[0])
    if entry <= 0:
        return None
    base_close = float(px['Close'].iloc[pos])
    hit10 = bool((fut['High'].max() / base_close - 1) >= 0.10) \
        if len(fut) >= 1 else None
    o = {
        'status': '확정' if len(fut) >= HOLD else f'진행중(D+{len(fut)})',
        'gap': entry / base_close - 1,
        'ret_now': float(fut['Close'].iloc[-1]) / entry - 1 - COST,
        'hit10_sofar': hit10,
        'days': len(fut),
    }
    return o


def run():
    if not PICKS_FILE.exists():
        print('picks_history.json 없음'); return
    hist = json.loads(PICKS_FILE.read_text(encoding='utf-8'))

    rows = []
    for day in hist:
        for p in day['picks']:
            o = outcome(p['code'], day['date'])
            if o is None:
                continue
            rows.append({'date': day['date'], 'rank': p['rank'],
                         'code': p['code'], 'name': p['name'],
                         'tradable': p.get('tradable', False), **o})
    if not rows:
        print('추적할 픽 없음'); return
    df = pd.DataFrame(rows)

    lines = ['# 픽 성과 추적', '',
             f'추적 픽 {len(df)}건 / {df["date"].nunique()}일치', '']

    # ── 매매픽 (규칙 대상) ──
    tr = df[df['tradable'] & (df['status'] != '진행중')]
    tr_done = tr[tr['status'] == '확정']
    lines.append('## 매매픽 (top5 & 비상한가 → 시가매수·5일보유)')
    if len(tr):
        cur = tr['ret_now']
        lines.append(f'- 진입 완료 {len(tr)}건 (확정 {len(tr_done)}건) | '
                     f'평균 수익 {cur.mean()*100:+.2f}% | '
                     f'승률 {(cur > 0).mean()*100:.0f}% | '
                     f'평균 갭 {tr["gap"].mean()*100:+.2f}%')
        for _, r in tr.iterrows():
            lines.append(f'  - {r["date"]} {r["name"]}({r["code"]}) '
                         f'[{r["status"]}] 갭 {r["gap"]*100:+.1f}% '
                         f'→ 현재 {r["ret_now"]*100:+.2f}%')
    else:
        lines.append('- 아직 없음')

    # ── 전체픽 예측 적중률 ──
    al = df[df['status'] != '진행중']
    lines.append('')
    lines.append('## 전체픽 예측 적중률 (5일 내 고가 +10%)')
    if len(al):
        top5 = al[al['rank'] <= 5]
        lines.append(f'- top5: {top5["hit10_sofar"].mean()*100:.0f}% '
                     f'({int(top5["hit10_sofar"].sum())}/{len(top5)}) | '
                     f'top20: {al["hit10_sofar"].mean()*100:.0f}% '
                     f'({int(al["hit10_sofar"].sum())}/{len(al)})')
    else:
        lines.append('- 윈도우 미성숙')

    report = '\n'.join(lines)
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    run()
