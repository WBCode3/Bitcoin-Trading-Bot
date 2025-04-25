from typing import Dict, Any
from datetime import timedelta

class TradingStrategy:
    def __init__(self):
        self.trailing_stop_active = False
        self.trailing_stop_price = 0.0
        self.last_optimization = None
        self.health_check_counter = 0
        self.last_signal = None
        self.last_signal_time = None
        self.min_signal_interval = timedelta(minutes=5)
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.position_size_multiplier = 1.0  # 풀시드 포지션
        self.leverage = 50  # 50배 레버리지
        self.max_leverage = 50  # 최대 레버리지
        self.min_leverage = 30  # 최소 레버리지
        self.liquidation_buffer = 0.05  # 청산가 버퍼 5%
        self.position_mode = 'hedge'

    def check_buy_condition(self, market_data: Dict[str, Any]) -> bool:
        """매수 조건 확인"""
        try:
            momentum = market_data.get('momentum', {})
            trend = market_data.get('trend', 'neutral')
            volatility = market_data.get('volatility', 'normal')
            
            # 모멘텀이 강하고 상승 추세일 때
            if momentum['state'] == 'strong' and trend == 'uptrend':
                return True
            
            # RSI가 과매도 구간이고 변동성이 낮을 때
            if momentum['rsi'] < 30 and volatility == 'low':
                return True
            
            # MACD 히스토그램이 양수이고 스토캐스틱이 상승 구간일 때
            if momentum['macd']['histogram'] > 0 and momentum['stochastic']['k'] > momentum['stochastic']['d']:
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"매수 조건 확인 중 오류 발생: {str(e)}")
            return False
        
    def check_sell_condition(self, market_data: Dict[str, Any]) -> bool:
        """매도 조건 확인"""
        try:
            momentum = market_data.get('momentum', {})
            trend = market_data.get('trend', 'neutral')
            volatility = market_data.get('volatility', 'normal')
            
            # 모멘텀이 약하고 하락 추세일 때
            if momentum['state'] == 'weak' and trend == 'downtrend':
                return True
            
            # RSI가 과매수 구간이고 변동성이 높을 때
            if momentum['rsi'] > 70 and volatility == 'high':
                return True
            
            # MACD 히스토그램이 음수이고 스토캐스틱이 하락 구간일 때
            if momentum['macd']['histogram'] < 0 and momentum['stochastic']['k'] < momentum['stochastic']['d']:
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"매도 조건 확인 중 오류 발생: {str(e)}")
            return False 