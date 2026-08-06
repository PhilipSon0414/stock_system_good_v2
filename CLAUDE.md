# stock_system_good_v2 — 세션 공통 지침

## 일일 자동 스캔 루틴의 이메일 요약 규칙

이 저장소의 일일 스캔 루틴(run_cloud_scan.sh 실행 후 한국어 요약 발송)은
스케줄 프롬프트의 ①~⑤ 항목에 **더해** 반드시 다음을 포함한다:

- **⑥ 매매픽별 매수/매도 코칭**: `reports/picks_<기준일>.md`의
  "매매 코칭" 섹션 내용을 요약해 픽마다 진입 조건(시가 갭 상한),
  익절 규칙(+10% 지정가), 손절선(D+1/D+3/D+5/D+7 체크포인트, 매수일=D0), 순항 기준
  (승자 경로 중앙값)을 전달한다. 사용자는 이 코칭대로 실제 매매한다.
- **⑦ 보유 포지션 신호**: `reports/positions_<기준일>.md`의 표와
  종목별 액션(🎯익절/⌛만기/✂️손절/⚠️주의/⏳보유)을 그대로 전달한다.
  손절·익절·만기 신호가 있으면 요약 첫 문장(폰 배너)에 올린다.

사용자의 매매 스타일: 1~2주 내 상승 예상 종목 매수 → +10% 도달 시 익절,
경로 이탈 시 조기 손절. 시스템의 o10 라벨·pattern_profile.json·
monitor_positions.py가 이 스타일을 구현한다 (README 3단계 참고).

사용자가 "OO 매수했어" 라고 알리면 `python3 monitor_positions.py add
<코드> <매수일> [매수가]`로 등록하고, "매도했어"면 `close <코드>` 처리.

## 개발 규약

- 산출물(reports/, picks_history.json, 모델 *.pkl, pattern_profile.json)은
  매 스캔 후 커밋하고 세션 지정 작업 브랜치에 푸시한다.
- `cache/`, `panel.pkl`, 자격증명(krx_config.json, email_config.json)은
  .gitignore — 커밋 금지.
- 주말·공휴일은 스캔 스킵 ("휴장일 — 스캔 스킵"으로만 요약).
