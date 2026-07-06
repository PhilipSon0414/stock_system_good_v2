#!/bin/bash
# v2 야간 파이프라인: 수급 갱신 → 예측 리포트 → 성과 추적
# launchd(com.stocksystem.v2_predict)가 평일 18:00 실행
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly_$(date +%Y%m%d).log"

cd "$DIR"
{
  echo "════ v2 nightly $(date '+%Y-%m-%d %H:%M:%S') ════"

  # 주말/공휴일 간단 가드: 토(6)/일(7)은 스킵
  dow=$(date +%u)
  if [ "$dow" -ge 6 ]; then
    echo "주말 — 스킵"
    exit 0
  fi

  echo "── 1/4 수급 갱신"
  python3 -W ignore fetch_flows.py update

  echo "── 2/4 예측 리포트 (가격 갱신 + 패널 + 스코어링 + v1 검증)"
  python3 -W ignore predict_today.py

  echo "── 3/4 성과 추적"
  python3 -W ignore track_performance.py

  echo "── 4/4 GitHub 동기화"
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "일일 자동 갱신: $(date '+%Y-%m-%d') 리포트·성과·모델"
    git push origin main || echo "⚠ push 실패 (다음 실행 때 재시도)"
  else
    echo "변경 없음 — 커밋 생략"
  fi

  echo "════ 완료 $(date '+%H:%M:%S') ════"
} >> "$LOG" 2>&1

# 30일 이상 된 로그 정리
find "$LOG_DIR" -name 'nightly_*.log' -mtime +30 -delete 2>/dev/null
exit 0
