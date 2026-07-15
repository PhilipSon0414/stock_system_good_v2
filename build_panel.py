# -*- coding: utf-8 -*-
"""
전시장 패널 데이터셋 구축 — 1단계

전 종목(KOSPI+KOSDAQ) × 과거 3년 일봉을 수집해
(종목, 날짜, D일까지의 피처, D+1~D+k 급등 라벨) 패널을 만든다.

데이터 소스: FinanceDataReader (네이버) — 종목당 ~0.1초, 캐시/재개 지원.
pykrx의 전종목-일자별 API는 KRX 로그인 요구로 차단돼 사용하지 않음.

사용법:
    python3 build_panel.py fetch      # 종목별 OHLCV 수집 (재개 가능)
    python3 build_panel.py build      # 피처+라벨 계산 → panel.pkl
    python3 build_panel.py all        # fetch + build
    python3 build_panel.py stats      # 패널 요약 통계

알려진 한계 (v1 범위):
  - 생존 편향: 현재 상장 종목 기준이라 최근 3년 내 상장폐지 종목 누락
  - 수급(외인/기관) 미포함: KRX 로그인 필요. v2에서 추가 예정
  - 시총은 '현재 주식수 × 과거 수정주가' 근사 (감자/증자 반영 안 됨)
"""
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config_v2 import (BASE_DIR, CACHE_DIR, PX_DIR, PANEL_FILE,
                       HISTORY_START, MIN_PRICE, MIN_MCAP,
                       SURGE_TH, HORIZONS)

LISTING_FILE = CACHE_DIR / 'krx_listing.pkl'
INDEX_FILE   = CACHE_DIR / 'index_ohlcv.pkl'


# ══════════════════════════════════════════════════════════════════════
# 1. 수집
# ══════════════════════════════════════════════════════════════════════

def _clean_universe(listing: pd.DataFrame) -> pd.DataFrame:
    """보통주만 남긴다 — 우선주/스팩/리츠/인프라 제외."""
    df = listing.copy()
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    df = df[df['Market'].isin(['KOSPI', 'KOSDAQ', 'KOSDAQ GLOBAL'])]
    df = df[df['Code'].str.endswith('0')]                      # 우선주 제외
    bad = df['Name'].str.contains('스팩|리츠|인프라|펀드|우$|우B$|우C$',
                                  regex=True, na=False)
    df = df[~bad]
    return df.reset_index(drop=True)


def fetch_listing() -> pd.DataFrame:
    import FinanceDataReader as fdr
    listing = fdr.StockListing('KRX')
    listing.to_pickle(LISTING_FILE)

    # 섹터 매핑 (실패해도 무방 — 테마 전염 피처는 시장 전체 기준으로 대체)
    try:
        desc = fdr.StockListing('KRX-DESC')
        desc.to_pickle(CACHE_DIR / 'krx_desc.pkl')
        print(f'  섹터 매핑 확보: {len(desc)}건')
    except Exception as e:
        print(f'  섹터 매핑 실패 (스킵): {e}')

    uni = _clean_universe(listing)
    print(f'  전체 {len(listing)} → 보통주 유니버스 {len(uni)}종목')
    return uni


def fetch_prices(uni: pd.DataFrame, refresh: bool = False):
    """종목별 일봉 캐시. 이미 받은 종목은 스킵 → 중단 후 재개 가능.
    refresh=True면 캐시 최종일이 오늘(리스팅 기준일)보다 뒤처진 종목만 재수집."""
    import FinanceDataReader as fdr
    codes = list(uni['Code'])
    done = fail = 0
    t0 = time.time()
    # 최신 거래일 기준: 삼성전자 캐시 없이 직접 조회 (주말/공휴일 안전)
    latest_td = None
    if refresh:
        try:
            latest_td = fdr.DataReader('005930', HISTORY_START).index[-1]
            print(f'  최신 거래일: {latest_td.date()}')
        except Exception:
            pass
    for i, code in enumerate(codes):
        out = PX_DIR / f'{code}.pkl'
        if out.exists():
            if not refresh or latest_td is None:
                done += 1
                continue
            try:   # 최신이면 스킵, 뒤처졌으면 재수집
                if pd.read_pickle(out).index[-1] >= latest_td:
                    done += 1
                    continue
            except Exception:
                pass
        try:
            df = fdr.DataReader(code, HISTORY_START)
            if df is not None and len(df) >= 60:      # 상장 60일 미만 제외
                df.to_pickle(out)
                done += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(codes)}  완료 {done} 실패 {fail}  ({el:.0f}s)',
                  flush=True)
        time.sleep(0.03)                               # 소스 서버 예의
    print(f'  수집 완료: {done}종목 (실패/제외 {fail})')

    # 시장 지수 (KOSPI/KOSDAQ) — market regime 피처용
    idx = {}
    for name, sym in [('kospi', 'KS11'), ('kosdaq', 'KQ11')]:
        try:
            idx[name] = fdr.DataReader(sym, HISTORY_START)
        except Exception as e:
            print(f'  지수 {sym} 실패: {e}')
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(idx, f)


# ══════════════════════════════════════════════════════════════════════
# 2. 종목별 피처 (모든 값은 당일 t 시점까지의 데이터만 사용)
# ══════════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, n=14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ticker_features(df: pd.DataFrame, code: str, shares: float,
                    flows: pd.DataFrame | None = None) -> pd.DataFrame:
    """일봉 → 피처 프레임. 인덱스=날짜.
    flows: fetch_flows.py 산출 (외인/기관 순매수 주수 + 외인 보유비율).
           없으면 수급 피처는 NaN (HistGBM이 네이티브 처리)."""
    o, h, l, c, v = (df['Open'].astype(float), df['High'].astype(float),
                     df['Low'].astype(float), df['Close'].astype(float),
                     df['Volume'].astype(float))
    out = pd.DataFrame(index=df.index)
    out['code']  = code
    out['close'] = c
    out['mcap']  = c * shares                     # 근사 시총

    # ── 수익률 ──
    out['ret1']  = c.pct_change(1)
    out['ret5']  = c.pct_change(5)
    out['ret20'] = c.pct_change(20)
    out['ret60'] = c.pct_change(60)

    # ── 거래량 ──
    vma30 = v.rolling(30).mean().shift(1)         # 당일 제외 30일 평균
    vr = v / vma30.replace(0, np.nan)
    out['vol_ratio'] = vr
    out['vol5_over20'] = (v.rolling(5).mean()
                          / v.rolling(20).mean().replace(0, np.nan))
    # 잠복(저거래) 연속일수 — 어제까지 기준
    quiet = (vr < 0.7)
    streak = quiet.groupby((~quiet).cumsum()).cumsum()   # True 연속 길이
    out['quiet_streak'] = streak.shift(1).fillna(0)
    out['vol_recovery'] = ((out['quiet_streak'] >= 3) & (vr >= 1.2)).astype(float)
    out['turnover'] = (c * v) / out['mcap'].replace(0, np.nan)   # 회전율 근사

    # ── 가격 구조 ──
    hh240 = h.rolling(240, min_periods=120).max()
    out['dist_52w_high'] = c / hh240 - 1
    out['dist_20d_high'] = c / h.rolling(20).max() - 1
    out['box20'] = ((h.rolling(20).max() - l.rolling(20).min()) / c)
    out['box5']  = ((h.rolling(5).max() - l.rolling(5).min()) / c)

    ma5, ma20, ma60, ma120 = (c.rolling(n).mean() for n in (5, 20, 60, 120))
    out['c_over_ma20']  = c / ma20 - 1
    out['ma5_over_ma20']  = ma5 / ma20 - 1
    out['ma20_over_ma60'] = ma20 / ma60 - 1
    out['ma_bull'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)
    out['c_over_ma120'] = c / ma120 - 1
    out['gc_gap'] = (ma60 - ma20) / ma60          # 양수=골든크로스 전 간격

    # ── 변동성/캔들 ──
    rng = (h - l)
    out['nr7'] = (rng == rng.rolling(7).min()).astype(float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
    out['atr_expand'] = atr14 / atr14.shift(20)
    out['volat20'] = out['ret1'].rolling(20).std()
    out['rsi14'] = _rsi(c)
    denom = rng.replace(0, np.nan)
    out['close_pos'] = (c - l) / denom            # 종가의 당일 범위 내 위치
    out['gap_open'] = o / c.shift(1) - 1

    # 연속 상승/하락일
    up = (out['ret1'] > 0)
    out['consec_up'] = up.groupby((~up).cumsum()).cumsum()

    # ── 라벨: 미래 k일 내 최고가 수익률 ──
    for k in HORIZONS:
        # 역방향 rolling max → 재반전 → -1 shift = max(High[t+1..t+k])
        fwd_max_high = h[::-1].rolling(k, min_periods=k).max()[::-1].shift(-1)
        ret = fwd_max_high / c - 1
        out[f'fwd_maxret_{k}d'] = ret
        out[f'label_{k}d'] = (ret >= SURGE_TH).astype(float).where(ret.notna())

    # ── 수급 (외인/기관 순매수) — 순매수 주수를 30일 평균 거래량으로 정규화
    #    ("며칠치 거래량을 순매수했나"). 데이터 없으면 NaN 유지.
    if flows is not None and len(flows):
        fl = flows.reindex(df.index)
        for who, col in [('frgn', 'Foreign'), ('inst', 'Inst')]:
            net = fl[col]
            out[f'{who}_net5']  = net.rolling(5,  min_periods=3).sum() / vma30
            out[f'{who}_net20'] = net.rolling(20, min_periods=10).sum() / vma30
            buy = (net > 0)
            out[f'{who}_consec_buy'] = (buy.groupby((~buy).cumsum())
                                        .cumsum().astype(float))
        out['frgn_ratio_chg5'] = fl['ForeignRatio'].diff(5)
    else:
        # 주의: 루프 변수로 'c'를 쓰면 종가 시리즈를 덮어써(섀도잉) 아래
        # 라벨(B) 계산이 문자열에 pct_change를 호출하며 전 종목 실패한다.
        for col in ('frgn_net5', 'frgn_net20', 'frgn_consec_buy',
                    'inst_net5', 'inst_net20', 'inst_consec_buy',
                    'frgn_ratio_chg5'):
            out[col] = np.nan

    # ── 라벨(B): v1 forward_tracker 호환 — k일 내 '단일일 종가 +10%' 발생 ──
    # (전일종가 대비 하루 등락률 기준. |±30%| 초과는 분할/권리락 아티팩트 제외)
    chg = c.pct_change(1)
    jump = ((chg >= SURGE_TH) & (chg.abs() <= 0.30)).astype(float)
    for k in HORIZONS:
        fut_jump = jump[::-1].rolling(k, min_periods=k).max()[::-1].shift(-1)
        out[f'label_jump_{k}d'] = fut_jump

    # ── 라벨(C): 실매매 기준 — D+1 '시가' 매수 후의 성과 ──
    # 시뮬레이션 결과 상위 픽의 평균 시가 갭이 +5~11%로 예측분을 선반영.
    # 종가 기준 라벨 대신 진입가(다음날 시가) 기준으로 직접 학습하기 위한 라벨.
    entry_open = o.shift(-1)                       # D+1 시가
    fwd_max_h5 = h[::-1].rolling(5, min_periods=5).max()[::-1].shift(-1)
    out['label_o5'] = ((fwd_max_h5 / entry_open - 1) >= SURGE_TH) \
        .astype(float).where(fwd_max_h5.notna() & entry_open.notna())
    # D+1 시가 → D+1 종가 +5% (당일 데이트레이드 기준)
    out['label_o1'] = ((c.shift(-1) / entry_open - 1) >= 0.05) \
        .astype(float).where(entry_open.notna() & c.shift(-1).notna())

    # 당일 급등 여부 (전염 피처 계산용 — 피처 아님)
    out['_surged_today'] = ((h / c.shift(1) - 1) >= SURGE_TH).astype(float)
    return out


# ══════════════════════════════════════════════════════════════════════
# 3. 패널 조립 (횡단면 피처 + 시장 피처)
# ══════════════════════════════════════════════════════════════════════

# 당일 전 종목 대비 백분위로 변환할 피처 — 절대값은 장세에 따라 의미가
# 변하므로(모두 뜨거운 날의 거래량 1.5x는 무의미) 횡단면 순위를 병행 학습
RANK_FEATS = ['vol_ratio', 'turnover', 'ret5', 'ret20', 'mcap', 'volat20']

def build_panel():
    listing = pd.read_pickle(LISTING_FILE)
    uni = _clean_universe(listing)
    shares_map = dict(zip(uni['Code'], uni['Stocks'].astype(float)))
    name_map   = dict(zip(uni['Code'], uni['Name']))

    # 섹터 매핑 (선택적)
    # KRX-DESC의 'Sector'는 소속부(벤처기업부 등) — 실제 업종은 'Industry'
    sector_map = {}
    desc_file = CACHE_DIR / 'krx_desc.pkl'
    if desc_file.exists():
        desc = pd.read_pickle(desc_file)
        if 'Industry' in desc.columns:
            desc['Code'] = desc['Code'].astype(str).str.zfill(6)
            sector_map = dict(zip(desc['Code'], desc['Industry']))

    frames = []
    files = sorted(PX_DIR.glob('*.pkl'))
    print(f'  피처 계산: {len(files)}종목')
    t0 = time.time()
    for i, f in enumerate(files):
        code = f.stem
        if code not in shares_map:
            continue
        try:
            px = pd.read_pickle(f)
            flow_f = CACHE_DIR / 'flow' / f'{code}.pkl'
            flows = pd.read_pickle(flow_f) if flow_f.exists() else None
            feat = ticker_features(px, code, shares_map[code], flows)
            frames.append(feat.astype({c: 'float32' for c in feat.columns
                                       if c != 'code'}, errors='ignore'))
        except Exception as e:
            print(f'  ⚠ {code}: {e}')
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(files)} ({time.time()-t0:.0f}s)', flush=True)

    panel = pd.concat(frames)
    panel.index.name = 'date'
    panel = panel.reset_index()
    panel['sector'] = panel['code'].map(sector_map).fillna('UNK')

    # ── 시장(날짜) 단위 피처 ──
    bydate = panel.groupby('date')
    mkt = pd.DataFrame({
        'mkt_up_ratio':    bydate.apply(lambda g: (g['ret1'] > 0).mean(),
                                        include_groups=False),
        'mkt_surge_cnt':   bydate['_surged_today'].sum(),
    })
    # 전염 피처는 '어제' 값 — 오늘 스캔 시점에 알 수 있는 정보만
    mkt = mkt.shift(1).rename(columns={'mkt_up_ratio': 'mkt_up_ratio_y',
                                       'mkt_surge_cnt': 'mkt_surge_cnt_y'})
    panel = panel.merge(mkt, on='date', how='left')

    # 지수 수익률
    if INDEX_FILE.exists():
        with open(INDEX_FILE, 'rb') as f:
            idx = pickle.load(f)
        for nm, df_i in idx.items():
            r = pd.DataFrame({
                f'{nm}_ret1': df_i['Close'].pct_change(1),
                f'{nm}_ret5': df_i['Close'].pct_change(5)})
            r.index.name = 'date'
            panel = panel.merge(r.reset_index(), on='date', how='left')

    # 섹터 전염: 같은 섹터에서 어제 급등이 나왔는가
    if sector_map:
        sec = (panel.groupby(['date', 'sector'])['_surged_today']
               .sum().rename('sec_surge_cnt').reset_index())
        sec = sec.sort_values('date')
        sec['sec_surge_cnt_y'] = (sec.groupby('sector')['sec_surge_cnt']
                                  .shift(1))
        panel = panel.merge(sec[['date', 'sector', 'sec_surge_cnt_y']],
                            on=['date', 'sector'], how='left')

    # 횡단면 백분위
    for col in RANK_FEATS:
        panel[f'{col}_pct'] = (panel.groupby('date')[col]
                               .rank(pct=True).astype('float32'))

    panel = panel.drop(columns=['_surged_today'])
    panel['name'] = panel['code'].map(name_map)

    panel.to_pickle(PANEL_FILE)
    print(f'  패널 저장: {PANEL_FILE}  ({len(panel):,}행, '
          f'{panel["date"].nunique()}거래일, {panel["code"].nunique()}종목)')
    return panel


def stats():
    panel = pd.read_pickle(PANEL_FILE)
    ok = panel[(panel['close'] >= MIN_PRICE) & (panel['mcap'] >= MIN_MCAP)]
    print(f'전체 {len(panel):,}행 → 필터 후 {len(ok):,}행 '
          f'(가격≥{MIN_PRICE}, 시총≥{MIN_MCAP/1e8:.0f}억)')
    for k in HORIZONS:
        lab = ok[f'label_{k}d'].dropna()
        print(f'  {k}일 내 고가 +{SURGE_TH*100:.0f}%: '
              f'기저율 {lab.mean()*100:.2f}%  (양성 {int(lab.sum()):,} / {len(lab):,})')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('fetch', 'all'):
        print('■ 1/2 종목 리스트 + 일봉 수집')
        uni = fetch_listing()
        fetch_prices(uni)
    if cmd in ('build', 'all'):
        print('■ 2/2 피처·라벨 계산 → 패널')
        build_panel()
        stats()
    if cmd == 'stats':
        stats()
