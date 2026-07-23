# -*- coding: utf-8 -*-
"""
stock_system_good_v2 설정

v1(stock_system)과의 핵심 차이:
  v1은 '프리필터를 통과한 후보(하루 ~83종목)'만 기록해 학습했다.
  → 내일 급등할 종목이 오늘 조용하면 학습 데이터에 아예 없음 (자체 진단:
    급등주 97.9%가 거래량 사전필터 탈락).
  v2는 전 종목 × 과거 3년 패널을 역추적해 '전체 시장에서 누가 급등하나'를
  직접 학습한다. 양성 사례 288건 → 수만 건.
"""
import os
from pathlib import Path

# SSL: 파이썬 3.14 macOS 프레임워크 빌드는 시스템 인증서 연결이 안 돼
# urllib 기반 요청(KRX 리스팅)이 실패한다. certifi 번들을 명시 지정.
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

BASE_DIR   = Path(__file__).parent

# KRX 로그인 (pykrx가 import 시 KRX_ID/KRX_PW 환경변수를 읽음)
# krx_config.json (권한 600, git 미추적)에 보관
try:
    import json as _json
    _krx = _json.loads((BASE_DIR / 'krx_config.json').read_text())
    os.environ.setdefault('KRX_ID', _krx['KRX_ID'])
    os.environ.setdefault('KRX_PW', _krx['KRX_PW'])
except Exception:
    pass
CACHE_DIR  = BASE_DIR / 'cache'
PX_DIR     = CACHE_DIR / 'px'          # 종목별 OHLCV 캐시
PANEL_FILE = BASE_DIR / 'panel.pkl'    # 최종 패널 데이터셋
MACRO_FILE = CACHE_DIR / 'macro.pkl'   # 매크로 선행지표 (달러/하이일드/구리)
CACHE_DIR.mkdir(exist_ok=True)
PX_DIR.mkdir(exist_ok=True)

# ── 수집 범위 ────────────────────────────────────────────────────────────
HISTORY_START = '2023-07-01'   # 3년치 역추적

# ── 유니버스 필터 (행 단위, 학습/평가 공통) ──────────────────────────────
MIN_PRICE  = 1_000             # 동전주 제외
MIN_MCAP   = 30e9              # 시총 300억 미만 초소형 제외 (조작성 급등 노이즈)

# 종목 자체 제외 규칙: 우선주(코드 끝자리 0 아님), 스팩, 리츠 등은
# build_panel.fetch 단계에서 이름/코드로 제거.

# ── 라벨 정의 ────────────────────────────────────────────────────────────
SURGE_TH = 0.10                # 급등 기준: +10%
HORIZONS = [1, 3, 5]           # D+1 / D+3 / D+5 내 최고가 기준
# 라벨: max(High[t+1..t+k]) / Close[t] - 1 >= SURGE_TH
# 종가 기준이 아닌 고가 기준 — 장중 10% 터치 후 밀린 케이스 포함

# ── 워크포워드 검증 ──────────────────────────────────────────────────────
EMBARGO_CAL_DAYS = 10          # 학습 꼬리 엠바고 (라벨 윈도우 경계 누수 차단)
N_TEST_MONTHS    = 12          # 최근 12개월 월 단위 워크포워드
TOP_KS           = [5, 10, 20] # precision@k 평가 지점
