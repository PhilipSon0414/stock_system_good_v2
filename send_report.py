# -*- coding: utf-8 -*-
"""
일일 리포트 이메일 발송 — v1의 email_config.json(Gmail 앱 비밀번호) 재사용

picks_history.json 최신일 픽 + performance.md 요약을 HTML 표로 발송.
run_nightly.sh 3.5단계에서 호출. 실패해도 파이프라인은 계속.

사용법: python3 send_report.py
"""
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from config_v2 import BASE_DIR

V1_EMAIL_CONFIG = BASE_DIR.parent / 'stock_system' / 'email_config.json'
PICKS_FILE      = BASE_DIR / 'picks_history.json'
PERF_FILE       = BASE_DIR / 'reports' / 'performance.md'


def _build_html(day: dict) -> str:
    rows = []
    for p in day['picks'][:10]:
        badge = ('<span style="color:#fff;background:#2e7d32;padding:1px 6px;'
                 'border-radius:3px;font-size:12px">매매</span>'
                 if p.get('tradable') else '')
        v1 = p.get('v1') or {}
        v1_txt = (f"세력{v1.get('seoryeok')}/T-{v1.get('tier')}"
                  if v1 else '—')
        color = '#c62828' if p['ret1'] > 0 else '#1565c0'
        rows.append(
            f"<tr><td>{p['rank']}</td><td>{badge}</td>"
            f"<td><b>{p['name']}</b> ({p['code']})</td>"
            f"<td align='right'>{p['close']:,.0f}</td>"
            f"<td align='right'><b>{p['prob_o5']*100:.1f}%</b></td>"
            f"<td align='right' style='color:{color}'>{p['ret1']*100:+.1f}%</td>"
            f"<td align='right'>{p['vol_ratio']:.1f}x</td>"
            f"<td>{v1_txt}</td></tr>")

    perf = ''
    if PERF_FILE.exists():
        perf_txt = PERF_FILE.read_text(encoding='utf-8')
        perf = ('<h3 style="margin-top:24px">픽 성과 추적</h3>'
                '<pre style="background:#f5f5f5;padding:10px;font-size:13px;'
                'white-space:pre-wrap">' + perf_txt.replace('#', '').strip()
                + '</pre>')

    return f"""
<div style="font-family:'Apple SD Gothic Neo',sans-serif;max-width:760px">
<h2>급등 후보 리포트 — 기준일 {day['date']}</h2>
<p style="color:#555;font-size:14px">
<b>매매 규칙</b>: <span style="color:#2e7d32">매매</span> 표시(전체 top5 &amp;
당일 +15% 미만 &amp; 5일 -30% 초과)만 <b>다음날 시가 매수 → 5거래일 보유</b>.
검증 성과 +1.27%/건·승률 45%. 나머지는 관찰용.</p>
<table border="0" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-size:14px;width:100%">
<tr style="background:#263238;color:#fff">
<th>#</th><th></th><th>종목</th><th>종가</th><th>o5확률</th>
<th>당일</th><th>거래량</th><th>v1 검증</th></tr>
{''.join(rows)}
</table>
{perf}
<p style="color:#999;font-size:12px;margin-top:20px">
stock_system_good_v2 자동 발송 · 상세: reports/picks_{day['date']}.md ·
GitHub: PhilipSon0414/stock_system_good_v2</p>
</div>"""


def send() -> bool:
    if not PICKS_FILE.exists():
        print('picks_history 없음'); return False
    hist = json.loads(PICKS_FILE.read_text(encoding='utf-8'))
    if not hist:
        print('픽 없음'); return False
    day = hist[-1]

    cfg = json.loads(V1_EMAIL_CONFIG.read_text(encoding='utf-8'))
    if not cfg.get('enabled'):
        print('이메일 비활성화 (v1 email_config.json)'); return False

    n_trade = sum(1 for p in day['picks'] if p.get('tradable'))
    msg = MIMEMultipart('alternative')
    msg['From'] = cfg['sender_email']
    msg['To'] = cfg['recipient_email']
    msg['Subject'] = (f"[v2 급등예측] {day['date']} — "
                      f"매매픽 {n_trade}건 / 후보 {len(day['picks'])}건")
    msg.attach(MIMEText(_build_html(day), 'html', 'utf-8'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(cfg['sender_email'],
                       cfg['app_password'].replace(' ', ''))
            smtp.sendmail(cfg['sender_email'], cfg['recipient_email'],
                          msg.as_string())
        print(f"이메일 발송 완료 → {cfg['recipient_email']}")
        return True
    except Exception as e:
        print(f'이메일 발송 실패: {e}')
        return False


if __name__ == '__main__':
    send()
