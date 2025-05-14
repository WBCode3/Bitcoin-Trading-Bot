import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 트레이딩 설정
TRADING_CONFIG = {
    'initial_capital': None,  # 실제 계좌 잔고 사용
    'leverage': 30,  # 레버리지 (30배)
    'fee': 0.00032,  # BNB 20% 할인 적용된 수수료
    'position_size': 0.5,  # 포지션 크기 (계좌의 50%)
    'stop_loss': 0.02,  # 손절 비율
    'take_profit': 0.04,  # 익절 비율
    'symbol': 'BTCUSDT'  # 거래 심볼 (BTC/USDT)
}

# 백테스트 설정
BACKTEST_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2023-12-31',
    'timeframe': '5m',  # 5분봉
}

# API 키 설정
API_CONFIG = {
    'binance': {
        'api_key': 'OO3jTAc6NkN7lGSXJElUs8mgpTvnnheDGUINbJZSYERefvsbM3Xl7cr9BaOiOwEb',
        'api_secret': 'DL8rz37aLlGCqgpLPksAZQMfGYbgyDG3J1TH8zXML1av8JJZCwRmJ713Y5qWmLoz'
    },
    'telegram': {
        'bot_token': '8131048961:AAEZWZJ9Op8DE1Z5fFfNg-zo94hsYvP_Yok',
        'chat_id': '5778241897'
    },
    'discord': {
        'webhook_url': 'https://discord.com/api/webhooks/1364400197689675826/4jox2qY2dEicafEsCM_A8Xtz_x9hd8xlU-lstKTizpRkDAkFEiED1UhEEaLsgkET792_'
    }
}

# 알림 설정
NOTIFICATION_CONFIG = {
    'telegram': {
        'enabled': True,
        'token': API_CONFIG['telegram']['bot_token'],
        'chat_id': API_CONFIG['telegram']['chat_id']
    },
    'discord': {
        'enabled': True,
        'webhook_url': API_CONFIG['discord']['webhook_url']
    }
}

# 설정 통합
settings = {
    'trading': TRADING_CONFIG,
    'backtest': BACKTEST_CONFIG,
    'notification': NOTIFICATION_CONFIG,
    'api': API_CONFIG
} 