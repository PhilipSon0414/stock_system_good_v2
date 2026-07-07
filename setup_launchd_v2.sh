#!/bin/bash
# v2 야간 파이프라인 LaunchAgent 등록 (평일 18:00)
# 사용법: bash setup_launchd_v2.sh          ← 등록
#         bash setup_launchd_v2.sh remove   ← 삭제
# 시각 근거: 장 마감 후 데이터 확정(~16시) 이후이면서
#            기존 v1 작업(19:00 model_retrain, 20:00 surge_learn,
#            20:10 daily_scan)과 겹치지 않는 18:00 선택.

LABEL="com.stocksystem.v2_predict"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PYTHON=$(which python3)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$1" = "remove" ] 2>/dev/null; then
    launchctl unload "$PLIST" 2>/dev/null
    rm -f "$PLIST"
    echo "LaunchAgent 제거 완료."
    exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"

cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${SCRIPT_DIR}/run_nightly.sh</string>
    </array>

    <!-- 18:00 본실행 + 20:30/22:30 캐치업 (run_nightly.sh가 중복 실행 스킵)
         Mac이 잠들어 트리거를 놓쳐도 깨어있는 첫 시점에 실행됨 -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>1</integer></dict>
        <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>2</integer></dict>
        <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>3</integer></dict>
        <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>4</integer></dict>
        <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>5</integer></dict>
        <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>1</integer></dict>
        <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>2</integer></dict>
        <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>3</integer></dict>
        <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>4</integer></dict>
        <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>5</integer></dict>
        <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>1</integer></dict>
        <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>2</integer></dict>
        <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>3</integer></dict>
        <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>4</integer></dict>
        <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>5</integer></dict>
    </array>

    <key>StandardOutPath</key>
    <string>${SCRIPT_DIR}/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>${SCRIPT_DIR}/logs/launchd_error.log</string>

    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>

    <key>RunAtLoad</key>
    <false/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$(dirname "$PYTHON"):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null
launchctl load "$PLIST"
echo "✅ 등록 완료 — 평일 18:00 자동 실행 (${LABEL})"
echo "   로그: ${SCRIPT_DIR}/logs/nightly_YYYYMMDD.log"
echo "   삭제: bash setup_launchd_v2.sh remove"
