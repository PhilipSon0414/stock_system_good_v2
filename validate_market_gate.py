# -*- coding: utf-8 -*-
"""
시장 과열 게이트 사후 검증 — '픽 당일 KOSPI 5일 수익률 > th → 매매 스킵'

simulate_picks.py 산출(sim_trades_label_o5_k5.json)에 KOSPI 지수를 붙여
게이트의 효과를 검증한다. 평균은 소수 대박이 견인하므로(아웃라이어 민감)
중앙값·승률·트림평균을 함께 본다. 임계값 민감도와 월별 일관성도 확인 —
특정 경계값에서만 좋아 보이면 과적합 신호다.

지수 소스: cache/index_ohlcv.pkl (build_panel 산출) 우선,
없으면 네이버 fchart API로 직접 수집 (pandas 불필요).

사용법: python3 validate_market_gate.py
출력:  reports/market_gate_validation.md + 콘솔
"""
import json

from config_v2 import BASE_DIR, CACHE_DIR, TRADABLE_RET1, OVERHEAT_R5

TRADES_FILE = BASE_DIR / 'sim_trades_label_o5_k5.json'
OUT_MD      = BASE_DIR / 'reports' / 'market_gate_validation.md'
COST        = 0.0025     # simulate_picks와 동일한 왕복 비용
THRESHOLDS  = [0.00, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]


def _avg(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def _med(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return float('nan')
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _trim(xs, p=0.02):
    """상하위 p씩 잘라낸 평균 — 대박/쪽박 아웃라이어 견고성 확인용."""
    s = sorted(xs)
    k = int(len(s) * p)
    return _avg(s[k:len(s) - k]) if len(s) > 2 * k else _avg(s)


def _kospi_r5_from_cache() -> dict | None:
    f = CACHE_DIR / 'index_ohlcv.pkl'
    if not f.exists():
        return None
    try:
        import pickle
        with open(f, 'rb') as fh:
            idx = pickle.load(fh)
        c = idx['kospi']['Close']
        r5 = c.pct_change(5).dropna()
        return {str(d.date()): float(v) for d, v in r5.items()}
    except Exception as e:
        print(f'  ⚠ 지수 캐시 실패 ({e}) — 네이버로 대체')
        return None


def _kospi_r5_from_naver() -> dict:
    import re
    import ssl
    import urllib.request
    url = ('https://fchart.stock.naver.com/sise.nhn'
           '?symbol=KOSPI&timeframe=day&count=600&requestType=0')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(
        req, context=ssl.create_default_context(), timeout=30
    ).read().decode('euc-kr', 'ignore')
    rows = [l.split('|') for l in re.findall(r'data="([^"]+)"', raw)]
    closes = [float(r[4]) for r in rows]
    out = {}
    for i, r in enumerate(rows):
        if i >= 5:
            d = r[0]
            out[f'{d[:4]}-{d[4:6]}-{d[6:]}'] = closes[i] / closes[i - 5] - 1
    return out


def run():
    trades = json.loads(TRADES_FILE.read_text(encoding='utf-8'))
    r5 = _kospi_r5_from_cache() or _kospi_r5_from_naver()

    # 실제 매매 규칙 거래셋: top5 중 픽 당일 +15% 미만 (사후 필터)
    R = [t for t in trades
         if t.get('ret1') is not None and t['ret1'] <= TRADABLE_RET1]
    no_idx = [t for t in R if t['date'] not in r5]
    R = [t for t in R if t['date'] in r5]

    lines = ['# 시장 과열 게이트 검증', '',
             f'거래셋: `{TRADES_FILE.name}` 중 매매규칙(픽 당일 ≤'
             f'{TRADABLE_RET1*100:.0f}%) {len(R)}건'
             + (f' (지수 미매칭 {len(no_idx)}건 제외)' if no_idx else ''),
             f'게이트: 픽 당일 KOSPI 5일 수익률 > th → 매매 스킵. '
             f'수익은 비용 {COST*100:.2f}% 차감 후 5일 보유(ret_5d) 기준.', '']

    # ── 임계값 민감도 ──
    lines += ['## 임계값 민감도', '',
              '| th | 유지 n | 평균/건 | 중앙값 | 트림평균(2%) | 승률 '
              '| 제외 n | 제외 평균 |', '|---|---|---|---|---|---|---|---|']
    for th in THRESHOLDS + [None]:
        keep = [t['ret_5d'] - COST for t in R
                if th is None or r5[t['date']] <= th]
        excl = [t['ret_5d'] - COST for t in R
                if th is not None and r5[t['date']] > th]
        if not keep:
            continue
        win = len([x for x in keep if x > 0]) / len(keep) * 100
        nm = f'+{th*100:.1f}%' if th is not None else '없음(전체)'
        ex = (f'{len(excl)} | {_avg(excl)*100:+.2f}%' if excl else '0 | —')
        star = ' ★' if th == OVERHEAT_R5 else ''
        lines.append(f'| {nm}{star} | {len(keep)} | {_avg(keep)*100:+.2f}% '
                     f'| {_med(keep)*100:+.2f}% | {_trim(keep)*100:+.2f}% '
                     f'| {win:.0f}% | {ex} |')
    lines += ['', f'★ = 운용 임계값 (`OVERHEAT_R5 = {OVERHEAT_R5}`). '
              '+1.5~+4% 구간에서 결과가 평탄하면 경계값 과적합이 아니라는 신호.', '']

    # ── 월별 일관성 ──
    months = sorted({t['month'] for t in R})
    lines += ['## 월별 일관성 (운용 임계값 기준)', '',
              '| 월 | 게이트 전 | n | 게이트 후 | n | 판정 |',
              '|---|---|---|---|---|---|']
    better = judged = 0
    for m in months:
        a = [t['ret_5d'] - COST for t in R if t['month'] == m]
        b = [t['ret_5d'] - COST for t in R
             if t['month'] == m and r5[t['date']] <= OVERHEAT_R5]
        if not b:
            lines.append(f'| {m} | {_avg(a)*100:+.2f}% | {len(a)} '
                         f'| 전량 스킵 | 0 | — |')
            continue
        d = _avg(b) - _avg(a)
        judged += 1
        better += d > 0
        lines.append(f'| {m} | {_avg(a)*100:+.2f}% | {len(a)} '
                     f'| {_avg(b)*100:+.2f}% | {len(b)} '
                     f'| {"개선" if d > 0 else "악화"} |')
    lines += ['', f'개선 월: **{better}/{judged}**', '',
              '## 한계', '',
              '- 단일 12개월 구간의 사후 분할 — 진짜 아웃오브샘플이 아니다.',
              '- 백테스트 기간 전체가 강세장. 하락장에서의 게이트 효과는 미검증.',
              '- 게이트가 유지하는 거래 수가 적은 달(과열 장기화)은 판정 신뢰도 낮음.']

    report = '\n'.join(lines)
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text(report, encoding='utf-8')
    print(report)
    print(f'\n저장: {OUT_MD}')


if __name__ == '__main__':
    run()
