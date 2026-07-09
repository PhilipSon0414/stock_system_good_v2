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

# ── 매매 규칙 (12개월 시뮬레이션 검증, simulate_picks / predict_today 공용) ──
TRADABLE_RET1 = 0.15   # 픽 당일 +15% 이상(상한가류) 제외 — 갭이 수익 잠식
KNIFE_RET5    = -0.30  # 5일 -30% 이하 폭락주(falling knife) 제외 — 백테스트 -4.2%/건

# 시장 과열 게이트: 픽 당일 KOSPI 5일 수익률 > +2%면 신규 매매 중단.
# 12개월 사후검증(validate_market_gate.py, 2026-07-09): 과열 구간 175건
# 평균 -2.06% → 제외 시 +1.04% → +2.77%/건, 승률 45→50%, 중앙값 -2.0→-0.3%,
# 2% 트림평균 -0.07→+1.27% (아웃라이어 승자에 의존하지 않는 개선).
# 임계값 +1.5~+4% 구간에서 결과 평탄 — 경계값에 민감하지 않음.
# 주의: 단일 12개월 구간의 사후 분할이라 과적합 가능성은 남음. 백테스트
# 기간 전체가 강세장(KOSPI 2,417→8,327)이었다는 한계도 동일.
OVERHEAT_R5 = 0.02

# ── 워크포워드 검증 ──────────────────────────────────────────────────────
EMBARGO_CAL_DAYS = 10          # 학습 꼬리 엠바고 (라벨 윈도우 경계 누수 차단)
N_TEST_MONTHS    = 12          # 최근 12개월 월 단위 워크포워드
TOP_KS           = [5, 10, 20] # precision@k 평가 지점
