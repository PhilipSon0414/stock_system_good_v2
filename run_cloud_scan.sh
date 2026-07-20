#!/bin/bash
# 클라우드 일일 스캔 — Claude Code Remote 루틴이 평일 15:35 KST에 실행
#
# Mac의 run_nightly.sh와 역할은 같지만 환경 차이를 반영:
#   - 컨테이너가 매번 새로 떠서 의존성 설치·전체 재수집이 필요 (병렬 12스레드)
#   - 수급(fetch_flows) 생략: 자체 검증상 무기여 + 시간 2배
#     (train_panel이 전부-NaN 피처를 자동 제외하므로 학습에 문제 없음)
#   - v1 정밀검증은 v1 저장소가 없어 predict_today가 자동 스킵
#   - SMTP 차단 환경이면 send_report는 HTML만 남기고 발송 생략
#     → 실제 이메일은 루틴의 완료 알림(요약 발송)이 담당
#   - git 커밋/푸시는 스크립트가 아니라 루틴 세션(에이전트)이 담당
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PY=python3

echo "════ cloud scan $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') ════"

# 주말 가드 (KST)
dow=$(TZ=Asia/Seoul date +%u)
if [ "$dow" -ge 6 ]; then
  echo "주말 — 스킵"
  exit 0
fi

echo "── 0/6 의존성"
$PY -c "import pandas, sklearn, joblib, FinanceDataReader" 2>/dev/null || \
  pip install --quiet "pandas<3" scikit-learn joblib finance-datareader certifi

# 오늘 종가가 소스(네이버)에 반영될 때까지 대기 (장 마감 15:30 직후 실행 대비,
# 최대 6회 × 5분). 공휴일이면 반영이 없으므로 30분 후 직전 거래일로 진행 —
# predict_today가 같은 기준일 리포트를 대체 기록하므로 무해.
TODAY=$(TZ=Asia/Seoul date +%Y-%m-%d)
for i in 1 2 3 4 5 6; do
  latest=$($PY -W ignore -c "
import FinanceDataReader as fdr
print(fdr.DataReader('005930', '2026-01-01').index[-1].date())" 2>/dev/null || true)
  if [ "$latest" = "$TODAY" ]; then
    echo "오늘($TODAY) 데이터 확인"
    break
  fi
  echo "최신 거래일 $latest ≠ $TODAY — 5분 대기 ($i/6)"
  sleep 300
done

echo "── 1/6 시세 수집 (병렬)"
FETCH_WORKERS=12 $PY -W ignore build_panel.py fetch

echo "── 2/6 패널 생성"
$PY -W ignore build_panel.py build

echo "── 3/6 모델 학습 (j1·5d·o5)"
for h in j1 5 o5; do
  $PY -W ignore train_panel.py train "$h" || echo "⚠ 학습 실패($h) — 기존 모델로 계속"
done

echo "── 4/6 예측 리포트"
$PY -W ignore predict_today.py --no-refresh

echo "── 5/6 성과 추적"
$PY -W ignore track_performance.py

echo "── 6/6 이메일 (SMTP 가능 환경에서만 실발송)"
$PY -W ignore send_report.py || echo "⚠ 이메일 발송 생략/실패 (루틴 알림이 대체)"

echo "════ 완료 $(TZ=Asia/Seoul date '+%H:%M:%S KST') ════"
