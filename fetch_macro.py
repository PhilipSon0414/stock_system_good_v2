# -*- coding: utf-8 -*-
"""
매크로 선행지표 수집 + 피처 생성 — 달러 / 하이일드 스프레드 / 구리

증시를 선제적으로 반영한다고 알려진 3계열을 패널 피처로 공급한다.

  1. 달러 인덱스 (DXY) + 원/달러 환율
     달러 강세 = 신흥국(한국) 자산에서 자금 이탈, 약세 전환 = 눌린 자산 반등.
     방향보다 '속도'가 신호 → 5/20일 변화율 + 가속도(변화율의 변화) 피처.
  2. 하이일드 스프레드 (ICE BofA US High Yield OAS, FRED)
     회사채-국채 금리차. 확대 = 위험회피, 축소 = 리스크온. 주가에 선행 경향.
     → 5/20일 변화 + 1년 z-score(공포 수준) 피처.
  3. 구리 선물 (HG) — '닥터 코퍼'
     실물경기 최선행 원자재. 바닥 반등 = 시장 반등 근접 신호.
     → 5/20일 수익률 + 60일 저점 대비 반등 플래그.

사전 검증 (2024-08~2026-06 공통 표본, KOSPI 미래 20일 수익률 대비 스피어만 IC):
  dxy_ret20  -0.13 (달러 급등 5분위 fwd20 +0.0% vs 나머지 +4~9%)
  hy_chg20   -0.19 (스프레드 급축소 5분위 +9.2% vs 급확대 +2.5%)
  hy_z252    +0.28 (스프레드 저점(안일) 5분위 -2.0% — 공포 정점이 오히려 기회)
  copper     +0.03 (약함), 바닥반등 플래그 +5.6% vs +4.4%
  단, 역방향(주가→HY) 상관도 존재 — 인과가 아닌 동행+선행 혼합. 판단은
  GBM이 다른 피처와의 상호작용으로 학습하게 두고, 여기선 원재료만 공급한다.

누수 차단 (핵심):
  미국 계열(DXY·HY·구리)의 날짜 t 데이터는 한국 시각 t+1 새벽에 확정된다.
  18:00 KST 스캔 시점에 쓸 수 있는 건 미국 전일(t-1)까지 → 한국 거래일
  캘린더로 reindex+ffill 후 1일 shift. FRED HY OAS는 발표가 하루 더 늦어
  (미 동부 아침 갱신) 2일 shift. 환율(USD/KRW)도 통일해서 1일 shift.

사용법:
    python3 fetch_macro.py fetch    # 수집 → cache/macro.pkl
    python3 fetch_macro.py check    # KOSPI 선행성 검증 리포트 (IC/5분위)
"""
import sys
import pickle

import numpy as np
import pandas as pd

from config_v2 import HISTORY_START, MACRO_FILE

# 계열별 후보 심볼 (앞에서부터 시도, 실패 시 폴백)
MACRO_SOURCES = {
    'dxy':    ['DX-Y.NYB'],                    # 달러 인덱스 (ICE)
    'usdkrw': ['USD/KRW'],                     # 원/달러 (달러 강세의 국내 직격탄)
    'hy':     ['FRED:BAMLH0A0HYM2'],           # 하이일드 OAS (%)
    'copper': ['HG=F', 'CPER'],                # 구리 선물 → 실패 시 구리 ETF
}
# 한국 거래일 기준 사용 가능 시점까지의 지연 (거래일): FRED 발표 지연 반영
MACRO_LAG = {'dxy': 1, 'usdkrw': 1, 'hy': 2, 'copper': 1}

MACRO_FEATS = ['dxy_ret5', 'dxy_ret20', 'dxy_accel',
               'usdkrw_ret5', 'usdkrw_ret20',
               'hy_chg5', 'hy_chg20', 'hy_z252',
               'copper_ret5', 'copper_ret20', 'copper_rebound']


def fetch_macro(start: str = None) -> dict:
    """계열별 종가 시리즈 수집 → MACRO_FILE 캐시. 실패 계열은 스킵(NaN 유지)."""
    import FinanceDataReader as fdr
    # z-score(252일)와 ret20 워밍업을 위해 패널 시작보다 1.5년 먼저
    start = start or str(pd.Timestamp(HISTORY_START) - pd.Timedelta(days=550))[:10]
    out = {}
    for name, symbols in MACRO_SOURCES.items():
        for sym in symbols:
            try:
                df = fdr.DataReader(sym, start)
                s = (df['Close'] if 'Close' in df.columns else df.iloc[:, 0])
                s = s.astype(float).dropna()
                if len(s) < 100:
                    raise ValueError(f'{len(s)}행 — 데이터 부족')
                out[name] = s
                print(f'  {name:8s} ← {sym:20s} {len(s)}행 '
                      f'(~{s.index[-1].date()})')
                break
            except Exception as e:
                print(f'  ⚠ {name} {sym} 실패: {e}')
    with open(MACRO_FILE, 'wb') as f:
        pickle.dump(out, f)
    print(f'  저장: {MACRO_FILE} ({len(out)}/{len(MACRO_SOURCES)}계열)')
    return out


def _series_features(raw: dict) -> pd.DataFrame:
    """원시 시리즈(각자의 캘린더) → 피처 프레임 (합집합 캘린더)."""
    feat = {}
    if 'dxy' in raw:
        d = raw['dxy']
        feat['dxy_ret5'] = d.pct_change(5)
        feat['dxy_ret20'] = d.pct_change(20)
        # '속도' 논지: 방향이 아니라 변화율의 변화 (꺾임 감지)
        feat['dxy_accel'] = d.pct_change(5) - d.pct_change(5).shift(5)
    if 'usdkrw' in raw:
        k = raw['usdkrw']
        feat['usdkrw_ret5'] = k.pct_change(5)
        feat['usdkrw_ret20'] = k.pct_change(20)
    if 'hy' in raw:
        h = raw['hy']
        feat['hy_chg5'] = h.diff(5)                      # %p 변화
        feat['hy_chg20'] = h.diff(20)
        feat['hy_z252'] = ((h - h.rolling(252, min_periods=120).mean())
                           / h.rolling(252, min_periods=120).std())
    if 'copper' in raw:
        c = raw['copper']
        feat['copper_ret5'] = c.pct_change(5)
        feat['copper_ret20'] = c.pct_change(20)
        # 바닥 반등: 60일 저점 대비 +5% 이상 & 20일선 위 (닥터 코퍼 반등 신호)
        feat['copper_rebound'] = (
            ((c / c.rolling(60).min() - 1 > 0.05)
             & (c > c.rolling(20).mean())).astype(float))
    return pd.DataFrame(feat)


def macro_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """한국 거래일 캘린더에 맞춘 누수-안전 매크로 피처.

    dates: 패널의 거래일 인덱스 (정렬됨). 반환: 같은 인덱스의 피처 프레임.
    캐시가 없거나 계열이 빠지면 해당 컬럼은 NaN (HistGBM 네이티브 처리).
    """
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    out = pd.DataFrame(index=dates, columns=MACRO_FEATS, dtype='float32')
    if not MACRO_FILE.exists():
        print('  ⚠ 매크로 캐시 없음 — 피처 NaN (fetch_macro.py fetch 먼저)')
        return out
    with open(MACRO_FILE, 'rb') as f:
        raw = pickle.load(f)
    feat = _series_features(raw)
    if feat.empty:
        return out
    # 자기 캘린더에서 피처 계산 완료 → 한국 거래일로 ffill 정렬 → 계열별 shift
    aligned = (feat.reindex(feat.index.union(dates)).ffill()
               .reindex(dates))
    for col in aligned.columns:
        series = col.split('_')[0]
        aligned[col] = aligned[col].shift(MACRO_LAG.get(series, 1))
    out.loc[:, aligned.columns] = aligned.astype('float32')
    return out


def check():
    """선행성 검증: 신호별 스피어만 IC + 5분위 미래 20일 KOSPI 수익률."""
    import FinanceDataReader as fdr
    if not MACRO_FILE.exists():
        fetch_macro()
    kospi = fdr.DataReader('KS11', HISTORY_START)['Close']
    feats = macro_features(kospi.index)
    fwd20 = kospi.pct_change(20).shift(-20)

    print(f'\n■ 매크로 신호 → KOSPI 미래 20일 수익률 '
          f'({kospi.index[0].date()}~, n={len(kospi)})')
    for col in MACRO_FEATS:
        s = feats[col]
        ok = s.notna() & fwd20.notna()
        if ok.sum() < 100 or s[ok].nunique() < 2:
            print(f'  {col:16s} 데이터 부족 — 스킵')
            continue
        ic = s[ok].corr(fwd20[ok], method='spearman')
        if s[ok].nunique() <= 2:                          # 플래그형
            m = fwd20[ok].groupby(s[ok]).mean() * 100
            desc = '  '.join(f'{int(k)}={v:+.2f}%' for k, v in m.items())
        else:
            q = pd.qcut(s[ok], 5, labels=False, duplicates='drop')
            m = fwd20[ok].groupby(q).mean() * 100
            desc = '  '.join(f'Q{int(k)+1} {v:+.2f}%' for k, v in m.items())
        print(f'  {col:16s} IC {ic:+.3f}   {desc}')
    print('\n  해석: IC 부호가 음(달러·스프레드 확대)/양(구리·z-score)이고 '
          '분위 간 차이가 크면 선행성 유효.')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'fetch'
    if cmd == 'fetch':
        print('■ 매크로 선행지표 수집 (달러/하이일드/구리)')
        fetch_macro()
    elif cmd == 'check':
        check()
