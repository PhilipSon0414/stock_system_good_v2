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

  echo "── 1/3 수급 갱신"
  python3 -W ignore fetch_flows.py update

  echo "── 2/3 예측 리포트 (가격 갱신 + 패널 + 스코어링 + v1 검증)"
  python3 -W ignore predict_today.py

  echo "── 3/3 성과 추적"
  python3 -W ignore track_performance.py

  echo "════ 완료 $(date '+%H:%M:%S') ════"
} >> "$LOG" 2>&1

# 30일 이상 된 로그 정리
find "$LOG_DIR" -name 'nightly_*.log' -mtime +30 -delete 2>/dev/null
exit 0
