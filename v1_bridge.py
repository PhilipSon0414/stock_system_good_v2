# -*- coding: utf-8 -*-
"""
v1 정밀 분석 브리지 — v2 매매픽을 v1(stock_system)의 세력/수급/티어로 2차 검증

v2 패널 모델 = 후보 생성기 (전 종목에서 확률 랭킹)
v1 analyze_one = 정밀 검증기 (세력 흔적·수급·ML 티어·진입 타이밍)

subprocess로 호출된다 (predict_today가 사용):
    python3 v1_bridge.py 005930 000660 ...
출력: JSON {code: {seoryeok, combined, tier, tier_label, entry_timing} | null}
      null = v1 세력 게이트 미달(세력 흔적 부족) 또는 분석 실패

주의: v1 모듈은 import 부작용(설정 로드 등)이 있어 별도 프로세스로 격리.
"""
import sys
import os
import json
from pathlib import Path

V1_DIR = Path(__file__).parent.parent / 'stock_system'


def main(codes):
    sys.path.insert(0, str(V1_DIR))
    os.chdir(V1_DIR)
    from daily_scan import analyze_one          # noqa: E402

    out = {}
    for code in codes:
        try:
            r = analyze_one(code)
        except Exception:
            r = None
        if r is None:
            out[code] = None                    # 게이트 미달/실패
            continue
        tier = r.get('ml_tier') or {}
        out[code] = {
            'seoryeok': r.get('seoryeok'),
            'surge':    r.get('surge'),
            'investor': r.get('investor'),
            'combined': r.get('combined'),
            'tier':     tier.get('tier'),
            'hit_rate': tier.get('hit_rate'),
            'entry_timing': r.get('entry_timing'),
        }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1:])
