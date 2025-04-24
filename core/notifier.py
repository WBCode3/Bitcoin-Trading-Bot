import logging
import requests
import os
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK')
        
    def send_telegram_message(self, message: str):
        """텔레그램으로 메시지 전송"""
        try:
            if self.telegram_token and self.telegram_chat_id:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                data = {
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                requests.post(url, data=data)
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 실패: {e}")
            
    def send_discord_message(self, message: str, embed: Dict[str, Any] = None):
        """디스코드로 메시지 전송"""
        try:
            if self.discord_webhook:
                data = {
                    "content": message,
                    "embeds": [embed] if embed else []
                }
                requests.post(self.discord_webhook, json=data)
        except Exception as e:
            logger.error(f"디스코드 메시지 전송 실패: {e}")
            
    def send_status_update(self, status: Dict[str, Any]):
        """상태 업데이트 메시지 전송"""
        try:
            # 텔레그램 메시지
            telegram_message = (
                f"<b>=== 트레이딩 봇 상태 업데이트 ===</b>\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"계좌 잔고: {status['balance']:,.2f} USDT\n"
                f"사용 가능: {status['free_balance']:,.2f} USDT\n"
                f"포지션: {status['position']}\n"
                f"일일 P&L: {status['daily_pnl']:.2%}\n"
                f"최대 드로다운: {status['max_drawdown']:.2%}\n"
                f"승률: {status['win_rate']:.2%}\n"
                f"최근 거래: {status['last_trade']}"
            )
            self.send_telegram_message(telegram_message)
            
            # 디스코드 메시지
            discord_embed = {
                "title": "트레이딩 봇 상태 업데이트",
                "color": 0x00ff00,
                "fields": [
                    {"name": "계좌 잔고", "value": f"{status['balance']:,.2f} USDT", "inline": True},
                    {"name": "사용 가능", "value": f"{status['free_balance']:,.2f} USDT", "inline": True},
                    {"name": "포지션", "value": str(status['position']), "inline": True},
                    {"name": "일일 P&L", "value": f"{status['daily_pnl']:.2%}", "inline": True},
                    {"name": "최대 드로다운", "value": f"{status['max_drawdown']:.2%}", "inline": True},
                    {"name": "승률", "value": f"{status['win_rate']:.2%}", "inline": True},
                    {"name": "최근 거래", "value": str(status['last_trade']), "inline": False}
                ],
                "timestamp": datetime.now().isoformat()
            }
            self.send_discord_message("", discord_embed)
            
        except Exception as e:
            logger.error(f"상태 업데이트 메시지 전송 실패: {e}")
            
    def send_trade_alert(self, trade: Dict[str, Any]):
        """거래 알림 메시지 전송"""
        try:
            # 텔레그램 메시지
            telegram_message = (
                f"<b>=== 새로운 거래 알림 ===</b>\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"종류: {trade['type']}\n"
                f"가격: {trade['price']:,.2f} USDT\n"
                f"수량: {trade['amount']}\n"
                f"레버리지: {trade['leverage']}x\n"
                f"사유: {trade['reason']}"
            )
            self.send_telegram_message(telegram_message)
            
            # 디스코드 메시지
            color = 0x00ff00 if trade['type'] == '매수' else 0xff0000
            discord_embed = {
                "title": f"새로운 {trade['type']} 거래",
                "color": color,
                "fields": [
                    {"name": "가격", "value": f"{trade['price']:,.2f} USDT", "inline": True},
                    {"name": "수량", "value": str(trade['amount']), "inline": True},
                    {"name": "레버리지", "value": f"{trade['leverage']}x", "inline": True},
                    {"name": "사유", "value": trade['reason'], "inline": False}
                ],
                "timestamp": datetime.now().isoformat()
            }
            self.send_discord_message("", discord_embed)
            
        except Exception as e:
            logger.error(f"거래 알림 메시지 전송 실패: {e}")
            
    def send_error_alert(self, error: str):
        """에러 알림 메시지 전송"""
        try:
            # 텔레그램 메시지
            telegram_message = (
                f"<b>⚠️ 에러 알림 ⚠️</b>\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"에러 내용: {error}"
            )
            self.send_telegram_message(telegram_message)
            
            # 디스코드 메시지
            discord_embed = {
                "title": "⚠️ 에러 알림",
                "color": 0xff0000,
                "description": error,
                "timestamp": datetime.now().isoformat()
            }
            self.send_discord_message("", discord_embed)
            
        except Exception as e:
            logger.error(f"에러 알림 메시지 전송 실패: {e}") 