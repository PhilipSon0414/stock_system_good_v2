# -*- coding: utf-8 -*-
"""
외국인/기관 수급 수집 — 다음 단계 ③

소스: 네이버 모바일 API (m.stock.naver.com/api/stock/{code}/trend)
  - pykrx의 KRX 엔드포인트는 로그인 요구로 차단 → v1과 같은 네이버 계열 사용
  - pageSize 최대 60, bizdate 커서로 과거 페이지네이션
  - 종목당 3년 ≈ 13요청 → 전 종목 ~33k요청 (4쓰레드, 재개 가능)

산출: cache/flow/{code}.pkl — index=날짜, cols=Foreign/Inst/Indiv(순매수 주수),
      ForeignRatio(외국인 보유비율 %)

사용법:
    python3 fetch_flows.py            # 전 종목 수집 (재개 가능)
    python3 fetch_flows.py update     # 일일 갱신 (최신 60일만 받아 병합)
    python3 fetch_flows.py 005930     # 단일 종목 테스트
"""
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from config_v2 import CACHE_DIR, PX_DIR, HISTORY_START

FLOW_DIR = CACHE_DIR / 'flow'
FLOW_DIR.mkdir(exist_ok=True)

_START = HISTORY_START.replace('-', '')
_UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}


def _get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _num(s):
    if not s or s in ('-', ''):
        return 0
    return int(str(s).replace(',', '').replace('+', ''))


def fetch_one(code: str) -> pd.DataFrame | None:
    rows = []
    cursor = None
    for _ in range(20):                        # 3년 = 13페이지, 여유 20
        url = (f'https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60'
               + (f'&bizdate={cursor}' if cursor else ''))
        d = _get(url)
        if not d:
            break
        rows += d
        last = d[-1]['bizdate']
        if last <= _START or len(d) < 60:
            break
        cursor = last
    if not rows:
        return None
    df = _parse_rows(rows)
    df = df[df.index >= HISTORY_START]
    return df


def fetch_all(workers: int = 4):
    codes = [f.stem for f in sorted(PX_DIR.glob('*.pkl'))]
    todo = [c for c in codes if not (FLOW_DIR / f'{c}.pkl').exists()]
    print(f'수급 수집: 전체 {len(codes)} / 남은 {len(todo)}종목 '
          f'({workers}쓰레드)')
    done = fail = 0
    t0 = time.time()

    def work(code):
        try:
            df = fetch_one(code)
            if df is not None and len(df):
                df.to_pickle(FLOW_DIR / f'{code}.pkl')
                return True
        except Exception:
            pass
        return False

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs)):
            if fu.result():
                done += 1
            else:
                fail += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(todo) - i - 1)
                print(f'  {i+1}/{len(todo)}  완료 {done} 실패 {fail}  '
                      f'({el:.0f}s, ETA {eta/60:.0f}분)', flush=True)
    print(f'수집 종료: 완료 {done} 실패 {fail}  ({(time.time()-t0)/60:.1f}분)')


def _parse_rows(rows) -> pd.DataFrame:
    df = pd.DataFrame([{
        'date': pd.to_datetime(r['bizdate']),
        'Foreign': _num(r.get('foreignerPureBuyQuant')),
        'Inst': _num(r.get('organPureBuyQuant')),
        'Indiv': _num(r.get('individualPureBuyQuant')),
        'ForeignRatio': float(str(r.get('foreignerHoldRatio', '0'))
                              .replace('%', '') or 0),
    } for r in rows])
    return df.drop_duplicates('date').set_index('date').sort_index()


def update_all(workers: int = 4):
    """일일 갱신: 최신 1페이지(60일)만 받아 기존 캐시에 병합.
    캐시 없는 신규 종목은 전체 수집."""
    codes = [f.stem for f in sorted(PX_DIR.glob('*.pkl'))]
    print(f'수급 일일 갱신: {len(codes)}종목 ({workers}쓰레드)')
    done = fail = 0
    t0 = time.time()

    def work(code):
        out = FLOW_DIR / f'{code}.pkl'
        try:
            if not out.exists():
                df = fetch_one(code)             # 신규 상장 등: 전체 수집
            else:
                rows = _get(f'https://m.stock.naver.com/api/stock/{code}'
                            f'/trend?pageSize=60')
                if not rows:
                    return False
                new = _parse_rows(rows)
                old = pd.read_pickle(out)
                df = pd.concat([old, new])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            if df is not None and len(df):
                df.to_pickle(out)
                return True
        except Exception:
            pass
        return False

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, c) for c in codes]
        for i, fu in enumerate(as_completed(futs)):
            if fu.result():
                done += 1
            else:
                fail += 1
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(codes)} ({time.time()-t0:.0f}s)',
                      flush=True)
    print(f'갱신 종료: 완료 {done} 실패 {fail}  ({(time.time()-t0)/60:.1f}분)')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        update_all()
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        df = fetch_one(sys.argv[1])
        print(df.tail(10) if df is not None else '실패')
    else:
        fetch_all()
