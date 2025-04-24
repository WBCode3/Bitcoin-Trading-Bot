import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Any

# .env 파일 로드
load_dotenv()

@dataclass
class TradingSettings:
    # 거래소 설정
    SYMBOL: str = "BTCUSDT"
    TIMEFRAME: str = "1m"
    LEVERAGE: int = 3
    
    # 리스크 관리 설정
    MAX_POSITION_SIZE: float = 0.1  # 최대 포지션 크기 (계좌의 %)
    STOP_LOSS_PCT: float = 0.02    # 손절 비율
    TAKE_PROFIT_PCT: float = 0.03  # 익절 비율
    DAILY_LOSS_LIMIT: float = 0.05 # 일일 손실 한도
    
    # 백테스트 설정
    BACKTEST_START: str = "2024-01-01"
    BACKTEST_END: str = "2024-02-01"
    INITIAL_BALANCE: float = 10000.0
    
    # 실행 모드
    RUN_BACKTEST: bool = True
    RUN_LIVE: bool = False
    
    # API 설정
    API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    API_SECRET: str = os.getenv('BINANCE_API_SECRET', '')
    
    # 알림 설정
    TELEGRAM_TOKEN: str = os.getenv('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    DISCORD_WEBHOOK: str = os.getenv('DISCORD_WEBHOOK', '')

# 설정 인스턴스 생성
settings = TradingSettings() 