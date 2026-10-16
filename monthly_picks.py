#!/usr/bin/env python3
"""월간/분기 실험 포트폴리오 + 모의매매 추적.

⚠ 실험용. 10년(2016~2026) 재검증에서 아래 후보 규칙은 2016~2023 구간에서
   초과수익이 없었다(20/40/60일: -0.6/+0.2/+0.8%p, 이긴기간 45~53%).
   2024~2026 구간의 우위(+2~9%p)는 국면 효과일 수 있다. → 실자금 투입 전
   이 스크립트의 paper_trades.json 모의 성과를 3개월 이상 확인할 것.
   10년 검증에서 일관되게 유효했던 것은 '회피 필터'뿐이다:
   60일 변동성/회전율/급등일 상위 10% 종목은 이후 20일 -1~2%p, 60일 -3~5%p 열위.

규칙(후보): 변동성 하위 50% & 52주 고점 -25% 이내 → 20일 평균 거래대금 큰 순 20종목
매매: 기준일 다음 거래일 시가 매수 → 40거래일 보유(20/60일 결과도 함께 기록)

사용:
  python3 monthly_picks.py            # 매월 27일 이후 스캔에서 발행 + 항상 정산
  python3 monthly_picks.py --force    # 날짜 무관 발행
  python3 monthly_picks.py --settle-only
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config_v2 import BASE_DIR, PX_DIR, PANEL_FILE, MIN_PRICE, MIN_MCAP

REPORT_DIR = BASE_DIR / 'reports'
REPORT_DIR.mkdir(exist_ok=True)
TRADES_FILE = BASE_DIR / 'paper_trades.json'
HORIZONS = (20, 40, 60)
HOLD = 40                  # 기본 보유일 (거래일)
N_PICKS = 20
COST = 0.006               # 왕복 비용(세금+수수료+슬리피지 가정)
MIN_VAL20 = 5e8            # 20일 평균 거래대금 5억
ISSUE_DAY = 27             # 매월 이 날짜 이후 첫 스캔부터 발행(같은 달 재발행은 대체)


def _features(asof):
    """기준일 종목별 피처. 기준일 데이터가 없는 종목(정지 등)은 제외."""
    panel = pd.read_pickle(PANEL_FILE)[['date', 'code', 'mcap', 'name']]
    last = panel[panel['date'] == panel['date'].max()].set_index('code')
    rows = []
    for f in PX_DIR.glob('*.pkl'):
        code = f.stem
        if code not in last.index:
            continue
        px = pd.read_pickle(f)
        if len(px) < 250 or px.index[-1] != asof:
            continue
        c = px['Close'].astype(float); h = px['High'].astype(float)
        v = px['Volume'].astype(float)
        r1 = c.pct_change()
        tv = (c * v).tail(20)
        rows.append({
            'code': code, 'name': last.at[code, 'name'], 'close': c.iloc[-1],
            'mcap': last.at[code, 'mcap'],
            'vol60': r1.tail(60).std(), 'maxup60': r1.tail(60).max(),
            'dhigh240': c.iloc[-1] / h.tail(240).max() - 1,
            'val20': tv.mean(), 'zeros20': int((v.tail(20) == 0).sum()),
            'turn_proxy': v.tail(20).mean() / max(v.tail(250).mean(), 1),
        })
    d = pd.DataFrame(rows)
    d = d[(d.close >= MIN_PRICE) & (d.mcap >= MIN_MCAP) &
          (d.val20 >= MIN_VAL20) & (d.zeros20 <= 2)].copy()
    for col in ('vol60', 'maxup60', 'turn_proxy'):
        d[col + '_p'] = d[col].rank(pct=True)
    return d


def _load():
    return json.loads(TRADES_FILE.read_text(encoding='utf-8')) \
        if TRADES_FILE.exists() else []


def issue(force=False):
    asof = pd.read_pickle(PX_DIR / '005930.pkl').index[-1]
    if not force and asof.day < ISSUE_DAY:
        print(f'  발행 생략 (기준일 {asof.date()} < 매월 {ISSUE_DAY}일)')
        return
    d = _features(asof)
    cand = (d[(d.vol60_p <= 0.5) & (d.dhigh240 > -0.25)]
            .sort_values('val20', ascending=False).head(N_PICKS))
    avoid = d[(d.vol60_p > 0.8) | (d.maxup60_p > 0.8) | (d.turn_proxy_p > 0.8)]
    month = asof.strftime('%Y-%m')
    # 같은 달 정규 발행(27일 이후)만 대체. 강제 발행분(--force)은 이미 매수일이 지났을 수 있어 보존
    trades = [t for t in _load()
              if t['asof'] != str(asof.date())
              and not (t['month'] == month and int(t['asof'][-2:]) >= ISSUE_DAY)]
    trades.append({
        'asof': str(asof.date()), 'month': month,
        'picks': [{'code': r.code, 'name': r['name'] if 'name' in r else '',
                   'close': float(r.close)} for _, r in cand.iterrows()],
        'universe': d['code'].tolist(), 'results': {}})
    TRADES_FILE.write_text(json.dumps(trades, ensure_ascii=False), encoding='utf-8')

    lines = [
        f'# 월간 실험 포트폴리오 — 기준일 {asof.date()}',
        '',
        '> ⚠ **실험용 · 검증 미통과.** 10년(2016~2026) 재검증에서 이 후보 규칙은 '
        '2016~2023 구간 초과수익이 없었습니다(20/40/60일: -0.6/+0.2/+0.8%p). '
        '2024~2026의 우위(+2~9%p)는 국면 효과일 수 있습니다. 실자금 투입 전 '
        '모의 성과(`reports/monthly_performance.md`)를 3개월 이상 확인하세요.',
        '',
        f'매매 방식: 다음 거래일 시가 매수 → **{HOLD}거래일** 보유 (20/60일 결과도 기록). '
        f'유니버스 {len(d):,}종목.',
        '',
        f'## 후보 {len(cand)}종목 (변동성 하위 50% · 52주 고점 -25% 이내 · 거래대금 큰 순)',
        '',
        '| # | 종목 | 코드 | 종가 | 20일 거래대금(억) | 60일 변동성 | 52주고점대비 |',
        '|---|---|---|---|---|---|---|']
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        lines.append(f"| {i} | {r['name']} | {r.code} | {r.close:,.0f} | "
                     f"{r.val20/1e8:,.0f} | {r.vol60*100:.1f}% | {r.dhigh240*100:+.1f}% |")
    lines += [
        '',
        f'## 회피 대상 (검증된 유일한 신호): {len(avoid)}종목 ({len(avoid)/len(d)*100:.0f}%)',
        '60일 변동성·회전율·급등일 중 하나라도 유니버스 상위 20%인 종목. 10년 검증에서 '
        '상위 10% 종목은 이후 20일 -1~2%p, 60일 -3~5%p 열위였습니다. 어떤 방식으로든 '
        '매수할 때는 이 종목들을 먼저 제외하는 것이 가장 일관된 개선이었습니다.']
    out = REPORT_DIR / f'monthly_{asof.date()}.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines)); print(f'\n저장: {out}')


def settle():
    trades = _load()
    if not trades:
        print('  모의매매 기록 없음'); return
    cache = {}
    def px_of(code):
        if code not in cache:
            try: cache[code] = pd.read_pickle(PX_DIR / f'{code}.pkl')
            except Exception: cache[code] = None
        return cache[code]
    def ret(code, asof, H):
        px = px_of(code)
        if px is None: return np.nan
        i = int(px.index.searchsorted(pd.Timestamp(asof), side='right'))  # 기준일 다음 거래일
        j = i + H - 1
        if j >= len(px) or i >= len(px): return np.nan
        o = float(px['Open'].iloc[i])
        return float(px['Close'].iloc[j]) / o - 1 if o > 0 else np.nan
    for t in trades:
        for H in HORIZONS:
            if str(H) in t['results']: continue
            pr = [ret(p['code'], t['asof'], H) for p in t['picks']]
            if any(np.isnan(x) for x in pr) or not pr: continue
            un = [x for x in (ret(c, t['asof'], H) for c in t['universe'])
                  if not np.isnan(x)]
            t['results'][str(H)] = {'pick': float(np.mean(pr)) - COST,
                                    'universe': float(np.mean(un)), 'n': len(pr)}
    TRADES_FILE.write_text(json.dumps(trades, ensure_ascii=False), encoding='utf-8')

    lines = ['# 월간 실험 포트폴리오 — 모의매매 성과', '',
             f'규칙 후보 {N_PICKS}종목 동일가중, 비용 {COST*100:.1f}% 차감, '
             '유니버스 = 같은 날 필터 통과 전 종목 동일가중.', '',
             '| 기준일 | 20일 | 40일 | 60일 |', '|---|---|---|---|']
    exs = {H: [] for H in HORIZONS}
    for t in trades:
        cells = []
        for H in HORIZONS:
            r = t['results'].get(str(H))
            if r:
                exs[H].append(r['pick'] - r['universe'])
                cells.append(f"{r['pick']*100:+.1f}% (유니버스 {r['universe']*100:+.1f}%)")
            else:
                cells.append('정산 대기')
        lines.append(f"| {t['asof']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 요약 (초과수익 = 포트폴리오 - 유니버스)']
    for H in HORIZONS:
        if exs[H]:
            e = np.array(exs[H])
            lines.append(f'- {H}일: {len(e)}건, 평균 초과 {e.mean()*100:+.1f}%p, '
                         f'이긴 횟수 {(e > 0).sum()}/{len(e)}')
        else:
            lines.append(f'- {H}일: 정산된 건 없음')
    lines.append('\n※ 표본이 쌓이기 전(최소 3~6건)에는 결론을 내리지 마세요.')
    (REPORT_DIR / 'monthly_performance.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    a = sys.argv[1:]
    if '--settle-only' not in a:
        issue(force='--force' in a)
    settle()
