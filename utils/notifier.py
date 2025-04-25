import requests
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class Notifier:
    def __init__(self):
        self.telegram_enabled = settings['notification']['telegram']['enabled']
        self.discord_enabled = settings['notification']['discord']['enabled']
        self.telegram_token = settings['notification']['telegram']['token']
        self.telegram_chat_id = settings['notification']['telegram']['chat_id']
        self.discord_webhook = settings['notification']['discord']['webhook_url']

    def send_message(self, message: str) -> None:
        """메시지 전송"""
        try:
            # 텔레그램 메시지
            if self.telegram_enabled:
                self._send_telegram(message)
            
            # 디스코드 메시지
            if self.discord_enabled:
                self._send_discord(message)
                
        except Exception as e:
            logger.error(f"메시지 전송 실패: {e}")

    def _send_telegram(self, message: str) -> None:
        """텔레그램 메시지 전송"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 실패: {e}")

    def _send_discord(self, message: str) -> None:
        """디스코드 메시지 전송"""
        try:
            data = {
                'content': message
            }
            response = requests.post(self.discord_webhook, json=data)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"디스코드 메시지 전송 실패: {e}") 