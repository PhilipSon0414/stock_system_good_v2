#!/bin/bash
# v2 야간 파이프라인: 수급 갱신 → 패널 → 모델 학습 → 스캔 → 성과 → 이메일 → GitHub
# launchd(com.stocksystem.v2_predict)가 평일 18:00 실행
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"

# 시스템 python3에는 pandas가 없음 — 프레임워크 파이썬 고정 (launchd 밖 수동 실행 대비)
PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
[ -x "$PY" ] || PY="python3"
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

  # 중복 실행 가드: 오늘 이미 완료했으면 스킵
  # (launchd 트리거가 18:00/20:30/22:30 3회라 — Mac이 잠들어 있어도
  #  깨어있는 첫 시점에 한 번만 실행되게 하는 캐치업 구조)
  if grep -q "════ 완료" "$LOG" 2>/dev/null; then
    echo "오늘 이미 완료 — 스킵 ($(date '+%H:%M'))"
    exit 0
  fi

  echo "── 1/7 수급 갱신"
  "$PY" -W ignore fetch_flows.py update

  echo "── 2/7 가격 갱신 + 패널 재생성"
  "$PY" -W ignore -c "
import build_panel as bp
bp.fetch_prices(bp.fetch_listing(), refresh=True)
bp.build_panel()"

  echo "── 3/7 모델 학습 (j1·5d·o5)"
  for h in j1 5 o5; do
    "$PY" -W ignore train_panel.py train "$h" || echo "⚠ 학습 실패($h) — 기존 모델로 계속"
  done

  echo "── 4/7 예측 리포트 (스코어링 + v1 검증)"
  "$PY" -W ignore predict_today.py --no-refresh

  echo "── 5/7 성과 추적"
  "$PY" -W ignore track_performance.py

  echo "── 6/7 이메일 발송"
  "$PY" -W ignore send_report.py || echo "⚠ 이메일 실패 (파이프라인 계속)"

  echo "── 7/7 GitHub 동기화"
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
