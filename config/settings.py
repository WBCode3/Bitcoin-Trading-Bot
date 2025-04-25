# 트레이딩 설정
TRADING_CONFIG = {
    'initial_capital': 500000,  # 초기 자본금
    'leverage': 50,  # 레버리지
    'fee': 0.00032,  # BNB 20% 할인 적용된 수수료
    'position_size': 1.0,  # 포지션 크기 (계좌의 %)
    'stop_loss': 0.02,  # 손절 비율
    'take_profit': 0.04,  # 익절 비율
}

# 백테스트 설정
BACKTEST_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2023-12-31',
    'timeframe': '5m',  # 5분봉
}

# 설정 통합
settings = {
    'trading': TRADING_CONFIG,
    'backtest': BACKTEST_CONFIG
} 