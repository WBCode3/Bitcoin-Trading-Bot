from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from config.settings import settings
from utils.logger import setup_logger
from .exchange import Exchange
from .risk_manager import RiskManager
from datetime import datetime, timedelta
import itertools
import logging
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class StrategyConfig:
    min_signal_interval: timedelta = timedelta(minutes=5)
    max_consecutive_losses: int = 5
    position_size_multiplier: float = 1.5
    leverage: int = 30
    max_leverage: int = 75
    min_leverage: int = 10
    liquidation_buffer: float = 0.03
    position_mode: str = 'hedge'

class TradingStrategy:
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.trailing_stop_active = False
        self.trailing_stop_price = 0.0
        self.last_optimization = None
        self.health_check_counter = 0
        self.last_signal = None
        self.last_signal_time = None
        self.consecutive_losses = 0
        self.logger = setup_logger('strategy')
        self._indicators_cache = {}
        self._last_calculation_time = None
        self.peak_profit = 0.0  # 최고 수익률 초기화
        self.first_target_hit = False  # 1차 청산 상태 초기화

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기술적 지표 계산 (캐싱 적용)"""
        try:
            current_time = datetime.now()
            cache_key = f"{df.index[-1]}_{len(df)}"
            
            # 캐시된 결과가 있고 5분 이내라면 재사용
            if (cache_key in self._indicators_cache and 
                self._last_calculation_time and 
                (current_time - self._last_calculation_time) < timedelta(minutes=5)):
                return self._indicators_cache[cache_key]

            indicators = self._calculate_all_indicators(df)
            self._indicators_cache[cache_key] = indicators
            self._last_calculation_time = current_time
            return indicators
            
        except Exception as e:
            self.logger.error(f"지표 계산 실패: {e}")
            return self._get_default_indicators()

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """모든 기술적 지표 계산"""
        # RSI
        rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        
        # 볼린저 밴드
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        
        # MACD
        macd_indicator = MACD(df['close'])
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_hist = macd_indicator.macd_diff().iloc[-1]
        
        # 스토캐스틱
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        stoch_k = stoch.stoch().iloc[-1]
        stoch_d = stoch.stoch_signal().iloc[-1]
        
        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        adx_value = adx.adx().iloc[-1]
        
        # 변동성 (ATR 기반)
        atr = df['high'] - df['low']
        volatility = atr.mean() / df['close'].mean()
        
        # 거래량
        volume = df['volume'].iloc[-1]
        volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
        
        return {
            'rsi': rsi,
            'bollinger': {
                'upper': bb_upper,
                'lower': bb_lower,
                'state': 'overbought' if df['close'].iloc[-1] > bb_upper else 'oversold' if df['close'].iloc[-1] < bb_lower else 'normal'
            },
            'macd': {
                'value': macd,
                'signal': macd_signal,
                'histogram': macd_hist,
                'state': 'bullish' if macd_hist > 0 else 'bearish'
            },
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'adx': adx_value,
            'volatility': volatility,
            'volume': {
                'current': volume,
                'ma': volume_ma,
                'ratio': volume_ratio
            }
        }

    def _get_default_indicators(self) -> Dict[str, Any]:
        """기본 지표 값 반환"""
        return {
            'rsi': 50,
            'bollinger': {'upper': 0, 'lower': 0, 'state': 'normal'},
            'macd': {'value': 0, 'signal': 0, 'histogram': 0, 'state': 'neutral'},
            'stoch_k': 50,
            'stoch_d': 50,
            'adx': 25,
            'volatility': 0.02,
            'volume': {'current': 0, 'ma': 0, 'ratio': 1.0}
        }

    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """매매 신호 생성"""
        try:
            if not self._validate_market_condition(data):
                return {'action': SignalType.HOLD.value, 'confidence': 0.0}

            indicators = data.get('indicators', {})
            trend = self._analyze_trend(indicators)
            
            # 매수 조건 확인
            if self._check_buy_conditions(indicators, trend):
                return {
                    'action': SignalType.BUY.value,
                    'confidence': self._calculate_confidence(indicators, trend),
                    'reason': self._get_buy_reason(data)
                }
            
            # 매도 조건 확인
            if self._check_sell_conditions(indicators, trend):
                return {
                    'action': SignalType.SELL.value,
                    'confidence': self._calculate_confidence(indicators, trend),
                    'reason': self._get_sell_reason(data)
                }
            
            return {'action': SignalType.HOLD.value, 'confidence': 0.0}
            
        except Exception as e:
            self.logger.error(f"신호 생성 중 오류 발생: {e}")
            return {'action': SignalType.HOLD.value, 'confidence': 0.0}

    def _calculate_confidence(self, indicators: Dict[str, Any], trend: Dict[str, Any]) -> float:
        """신호 신뢰도 계산"""
        confidence = 0.0
        
        # RSI 기반 신뢰도
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            confidence += 0.2
        
        # MACD 기반 신뢰도
        macd = indicators.get('macd', {})
        if macd.get('state') == 'bullish' and macd.get('histogram', 0) > 0:
            confidence += 0.3
        
        # ADX 기반 신뢰도
        adx = indicators.get('adx', 25)
        if adx > 25:
            confidence += 0.2
        
        # 볼린저 밴드 기반 신뢰도
        bb = indicators.get('bollinger', {})
        if bb.get('state') == 'normal':
            confidence += 0.3
        
        return min(confidence, 1.0)

    def _validate_market_condition(self, data: Dict[str, Any]) -> bool:
        """시장 조건 검증"""
        try:
            if not data or not isinstance(data, dict):
                return False
            
            df = data.get('df')
            if df is None or not isinstance(df, pd.DataFrame):
                return False
            
            if len(df) < 100:  # 최소 데이터 포인트
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"시장 조건 검증 중 오류 발생: {e}")
            return False

    def _check_buy_conditions(self, indicators: Dict[str, Any], trend: Dict[str, Any]) -> bool:
        """매수 조건 확인"""
        try:
            # RSI 조건
            rsi = indicators.get('rsi', 50)
            if rsi > 70:  # 과매수 구간
                return False
            
            # MACD 조건
            macd = indicators.get('macd', {})
            if not (macd.get('state') == 'bullish' and macd.get('histogram', 0) > 0):
                return False
            
            # 볼린저 밴드 조건
            bb = indicators.get('bollinger', {})
            if bb.get('state') != 'oversold':
                return False
            
            # ADX 조건
            adx = indicators.get('adx', 25)
            if adx < 25:  # 추세가 약함
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"매수 조건 확인 중 오류 발생: {e}")
            return False

    def _check_sell_conditions(self, indicators: Dict[str, Any], trend: Dict[str, Any]) -> bool:
        """매도 조건 확인"""
        try:
            # RSI 조건
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # 과매도 구간
                return False
            
            # MACD 조건
            macd = indicators.get('macd', {})
            if not (macd.get('state') == 'bearish' and macd.get('histogram', 0) < 0):
                return False
            
            # 볼린저 밴드 조건
            bb = indicators.get('bollinger', {})
            if bb.get('state') != 'overbought':
                return False
            
            # ADX 조건
            adx = indicators.get('adx', 25)
            if adx < 25:  # 추세가 약함
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"매도 조건 확인 중 오류 발생: {e}")
            return False

    def _get_buy_reason(self, market_data: Dict[str, Any]) -> str:
        """매수 신호 사유"""
        indicators = market_data.get('indicators', {})
        reasons = []
        
        if indicators.get('rsi', 50) < 30:
            reasons.append("RSI 과매도")
        if indicators.get('macd', {}).get('state') == 'bullish':
            reasons.append("MACD 상승 전환")
        if indicators.get('bollinger', {}).get('state') == 'oversold':
            reasons.append("볼린저 밴드 하단 터치")
        
        return ", ".join(reasons) if reasons else "기술적 지표 조합"

    def _get_sell_reason(self, market_data: Dict[str, Any]) -> str:
        """매도 신호 사유"""
        indicators = market_data.get('indicators', {})
        reasons = []
        
        if indicators.get('rsi', 50) > 70:
            reasons.append("RSI 과매수")
        if indicators.get('macd', {}).get('state') == 'bearish':
            reasons.append("MACD 하락 전환")
        if indicators.get('bollinger', {}).get('state') == 'overbought':
            reasons.append("볼린저 밴드 상단 터치")
        
        return ", ".join(reasons) if reasons else "기술적 지표 조합"

    def update_trade_result(self, is_profit: bool):
        """거래 결과 업데이트"""
        if is_profit:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def reset(self):
        """전략 초기화"""
        self.trailing_stop_active = False
        self.trailing_stop_price = 0.0
        self.last_signal = None
        self.last_signal_time = None
        self.consecutive_losses = 0
        self._indicators_cache.clear()
        self._last_calculation_time = None
        self.peak_profit = 0.0  # 최고 수익률 초기화
        self.first_target_hit = False  # 1차 청산 상태 초기화

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """피보나치 레벨 계산"""
        try:
            high = df['high'].max()
            low = df['low'].min()
            diff = high - low
            
            levels = {
                '0.236': high - diff * 0.236,
                '0.382': high - diff * 0.382,
                '0.5': high - diff * 0.5,
                '0.618': high - diff * 0.618,
                '0.786': high - diff * 0.786
            }
            
            return levels
            
        except Exception as e:
            logger.error(f"피보나치 레벨 계산 실패: {e}")
            return {}

    def detect_crash(self, df15: pd.DataFrame, df1h: pd.DataFrame) -> bool:
        """크래시 감지 (15분 타임프레임용)"""
        try:
            # 15분 전 대비 가격 변화
            price_change = (df15['close'].iloc[-1] - df15['close'].iloc[-2]) / df15['close'].iloc[-2]
            
            # RSI 변화
            rsi15 = RSIIndicator(df15['close']).rsi()
            rsi_change = abs(rsi15.iloc[-1] - rsi15.iloc[-2])
            
            # MACD 음전환
            macd15 = MACD(df15['close']).macd_diff()
            macd_signal = macd15.iloc[-1] < 0 and macd15.iloc[-2] > 0
            
            return (price_change <= -0.02 and rsi_change > 15 and macd_signal)
            
        except Exception as e:
            logger.error(f"크래시 감지 실패: {e}")
            return False

    def detect_spike(self, df15: pd.DataFrame) -> bool:
        """스파이크 감지 (15분 타임프레임용)"""
        try:
            # 30분 전 대비 가격 변화
            price_change = (df15['close'].iloc[-1] - df15['close'].iloc[-2]) / df15['close'].iloc[-2]
            
            # RSI
            rsi = RSIIndicator(df15['close']).rsi()
            
            # MACD 양전환
            macd = MACD(df15['close']).macd_diff()
            macd_signal = macd.iloc[-1] > 0 and macd.iloc[-2] < 0
            
            return (price_change >= 0.03 and rsi.iloc[-1] > 75 and macd_signal)
            
        except Exception as e:
            logger.error(f"스파이크 감지 실패: {e}")
            return False

    def check_entry_conditions(self, indicators: Dict[str, Any]) -> bool:
        """진입 조건 확인"""
        try:
            # 1. ADX 기반 추세 확인 (ADX >= 25)
            adx = indicators['adx'].iloc[-1]
            if adx < 25:
                self.logger.info(f"ADX {adx:.2f} < 25: 추세가 약함")
                return False
                
            # 2. RSI + MACD 조합으로 방향 결정
            rsi = indicators['rsi'].iloc[-1]
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            macd_hist = indicators['macd_hist'].iloc[-1]
            
            # 롱 포지션 조건
            if (rsi < 30 and  # RSI 과매도
                macd > macd_signal and  # MACD 상향 돌파
                macd_hist > 0):  # MACD 히스토그램 양수
                
                self.logger.info(f"롱 진입 신호 발생:")
                self.logger.info(f"- ADX: {adx:.2f}")
                self.logger.info(f"- RSI: {rsi:.2f}")
                self.logger.info(f"- MACD: {macd:.2f} > Signal: {macd_signal:.2f}")
                return True
                
            # 숏 포지션 조건
            if (rsi > 70 and  # RSI 과매수
                macd < macd_signal and  # MACD 하향 돌파
                macd_hist < 0):  # MACD 히스토그램 음수
                
                self.logger.info(f"숏 진입 신호 발생:")
                self.logger.info(f"- ADX: {adx:.2f}")
                self.logger.info(f"- RSI: {rsi:.2f}")
                self.logger.info(f"- MACD: {macd:.2f} < Signal: {macd_signal:.2f}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"진입 조건 확인 중 오류 발생: {e}")
            return False

    def _adjust_leverage(self, market_data: Dict[str, Any]) -> None:
        """ATR 기반 레버리지 조정"""
        try:
            # ATR 데이터 수집 (1시간봉 기준)
            atr_series = market_data['indicators']['atr_series']
            current_price = market_data['current_price']
            
            # 최소 데이터 포인트 확인
            if len(atr_series) < 14:
                self.logger.warning("ATR 데이터 부족: 기본 레버리지 사용")
                self.config.leverage = 10
                return
            
            # 변동성 계산 (σ = ATR / Price)
            current_atr = atr_series.iloc[-1]
            volatility = current_atr / current_price
            
            # 레버리지 계산: leverage = max(10, min(75, int((1.8/σ) + 5)))
            calculated_leverage = int((1.8 / volatility) + 5)
            self.config.leverage = max(10, min(75, calculated_leverage))
            
            # 상세 로깅
            self.logger.info(f"=== 레버리지 조정 정보 ===")
            self.logger.info(f"현재 ATR14: {current_atr:.4f}")
            self.logger.info(f"현재 가격: {current_price:.2f}")
            self.logger.info(f"변동성(σ): {volatility:.4f}")
            self.logger.info(f"계산된 레버리지: {calculated_leverage}x")
            self.logger.info(f"최종 적용 레버리지: {self.config.leverage}x")
            self.logger.info(f"=====================")
            
        except Exception as e:
            self.logger.error(f"레버리지 조정 중 오류 발생: {e}")
            # 오류 발생 시 안전한 기본값 사용
            self.config.leverage = 10

    def check_stop_loss(self, pnl: float, volatility: float, side: str, entry_price: float, current_price: float) -> bool:
        """손절 조건 확인"""
        try:
            # 기본 손절 비율
            base_sl_pct = 1.0  # 1%
            
            # 레버리지 기반 조정 계수
            adj = 1 / (self.config.leverage ** 0.5)  # sqrt(leverage)로 나누기
            
            # 최종 손절 비율 계산
            sl_pct = base_sl_pct * adj
            
            # 손절 조건 확인
            if side == 'long':
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct > sl_pct:
                    self.logger.info(f"손절 조건 충족: 손실률 {loss_pct:.2%} > {sl_pct:.2%}")
                    return True
            else:  # short
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct > sl_pct:
                    self.logger.info(f"손절 조건 충족: 손실률 {loss_pct:.2%} > {sl_pct:.2%}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"손절 조건 확인 중 오류: {e}")
            return False

    def check_take_profit(self, pnl: float, volatility: float, side: str, entry_price: float, current_price: float) -> Tuple[bool, float]:
        """익절 조건 확인"""
        try:
            # 기본 익절 비율
            base_tp1_pct = 1.5  # 1.5%
            base_tp2_pct = 3.0  # 3.0%
            
            # 레버리지 기반 조정 계수
            adj = 1 / (self.config.leverage ** 0.5)
            
            # 최종 익절 비율 계산
            tp1_pct = base_tp1_pct * adj
            tp2_pct = base_tp2_pct * adj
            
            # 현재 수익률 계산
            if side == 'long':
                profit_pct = (current_price - entry_price) / entry_price
            else:  # short
                profit_pct = (entry_price - current_price) / entry_price
            
            # 최고 수익률 추적
            if not hasattr(self, 'peak_profit'):
                self.peak_profit = profit_pct
            else:
                self.peak_profit = max(self.peak_profit, profit_pct)
            
            # 트레일링 스탑 로직
            if self.peak_profit > tp1_pct:
                # 1차 청산 (최고 수익의 30% 하락 시)
                if not hasattr(self, 'first_target_hit') and profit_pct <= self.peak_profit * 0.7:
                    self.logger.info(f"1차 익절 조건 충족:")
                    self.logger.info(f"- 최고 수익률: {self.peak_profit:.2%}")
                    self.logger.info(f"- 현재 수익률: {profit_pct:.2%}")
                    self.logger.info(f"- 하락률: {(1 - profit_pct/self.peak_profit):.2%}")
                    self.first_target_hit = True
                    return True, 0.5  # 50% 청산
                
                # 2차 청산 (최고 수익의 60% 하락 시)
                if hasattr(self, 'first_target_hit') and profit_pct <= self.peak_profit * 0.4:
                    self.logger.info(f"2차 익절 조건 충족:")
                    self.logger.info(f"- 최고 수익률: {self.peak_profit:.2%}")
                    self.logger.info(f"- 현재 수익률: {profit_pct:.2%}")
                    self.logger.info(f"- 하락률: {(1 - profit_pct/self.peak_profit):.2%}")
                    return True, 1.0  # 잔여분 전량 청산
            
            # 초기 익절 조건 (트레일링 스탑 활성화 전)
            if profit_pct >= tp2_pct:
                self.logger.info(f"초기 익절 조건 충족: 수익률 {profit_pct:.2%} >= {tp2_pct:.2%}")
                return True, 1.0  # 전량 청산
            
            return False, 0.0
            
        except Exception as e:
            self.logger.error(f"익절 조건 확인 중 오류: {e}")
            return False, 0.0

    def _calculate_liquidation_price(self, side: str, entry_price: float, leverage: float) -> float:
        """청산가 계산"""
        try:
            # 바이낸스 선물 청산가 계산식
            if side == 'long':
                return entry_price * (1 - 1/leverage)
            else:
                return entry_price * (1 + 1/leverage)
        except Exception as e:
            logger.error(f"청산가 계산 실패: {e}")
            return 0.0

    def check_exit_conditions(self, indicators: Dict[str, Any]) -> bool:
        """청산 조건 확인"""
        try:
            # 1. ADX 기반 추세 약화 확인
            adx = indicators['adx'].iloc[-1]
            if adx < 20:  # 추세가 약해지면 청산
                self.logger.info(f"ADX {adx:.2f} < 20: 추세 약화로 청산")
                return True
                
            # 2. RSI + MACD 반전 신호
            rsi = indicators['rsi'].iloc[-1]
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            
            # 롱 포지션 청산
            if (rsi > 70 or  # RSI 과매수
                (macd < macd_signal and macd_hist < 0)):  # MACD 하향 돌파
                self.logger.info(f"롱 청산 신호 발생:")
                self.logger.info(f"- RSI: {rsi:.2f}")
                self.logger.info(f"- MACD: {macd:.2f} < Signal: {macd_signal:.2f}")
                return True
                
            # 숏 포지션 청산
            if (rsi < 30 or  # RSI 과매도
                (macd > macd_signal and macd_hist > 0)):  # MACD 상향 돌파
                self.logger.info(f"숏 청산 신호 발생:")
                self.logger.info(f"- RSI: {rsi:.2f}")
                self.logger.info(f"- MACD: {macd:.2f} > Signal: {macd_signal:.2f}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"청산 조건 확인 중 오류 발생: {e}")
            return False

    def execute_trade(self, market_data: Dict[str, Any]) -> None:
        """거래 실행"""
        try:
            # 최종 검증
            if not self._validate_trade_conditions(market_data):
                self.logger.warning("거래 조건 검증 실패: 거래 취소")
                return

            # 포지션 모드 확인 및 설정
            if not self._check_and_set_position_mode():
                self.logger.error("포지션 모드 설정 실패")
                return

            # 매수 수량 계산
            position_size = self.calculate_position_size(market_data['current_price'], market_data['indicators']['atr'].iloc[-1])
            
            if position_size <= 0:
                self.logger.error("유효하지 않은 포지션 사이즈")
                return
            
            # 시장가 매수 주문
            order = self.exchange.create_order(
                symbol=self.exchange.symbol,
                side='BUY',
                quantity=position_size,
                order_type='MARKET'
            )
            
            if not order:
                self.logger.error("주문 실행 실패")
                return
            
            # 5분 대기
            time.sleep(300)
            
            # 시장가 매도 주문
            exit_order = self.exchange.create_order(
                symbol=self.exchange.symbol,
                side='SELL',
                quantity=position_size,
                order_type='MARKET'
            )
            
            if not exit_order:
                self.logger.error("청산 주문 실행 실패")
                # 여기서 추가적인 청산 시도나 알림 로직 추가 가능
            
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {e}")
            # 오류 발생 시 안전한 상태로 복구
            self._recover_from_error()
            return

    def _validate_trade_conditions(self, market_data: Dict[str, Any]) -> bool:
        """거래 실행 전 최종 검증"""
        try:
            # 기본 데이터 검증
            if not market_data or not isinstance(market_data, dict):
                return False

            # 가격 데이터 검증
            if 'current_price' not in market_data or market_data['current_price'] <= 0:
                return False

            # 지표 데이터 검증
            indicators = market_data.get('indicators', {})
            if not indicators or not isinstance(indicators, dict):
                return False

            # ATR 데이터 검증
            if 'atr' not in indicators or len(indicators['atr']) < 14:
                return False

            # 거래량 데이터 검증
            if 'volume' not in indicators:
                return False

            return True

        except Exception as e:
            self.logger.error(f"거래 조건 검증 중 오류 발생: {e}")
            return False

    def _check_and_set_position_mode(self) -> bool:
        """포지션 모드 확인 및 설정"""
        try:
            current_mode = self.exchange.get_position_mode()
            if current_mode != self.config.position_mode:
                success = self.exchange.set_position_mode(self.config.position_mode)
                if not success:
                    self.logger.error(f"포지션 모드 변경 실패: {self.config.position_mode}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"포지션 모드 확인 중 오류 발생: {e}")
            return False

    def _recover_from_error(self):
        """오류 발생 시 복구"""
        try:
            # 현재 포지션 확인
            position = self.exchange.get_position()
            if position and position['size'] > 0:
                # 포지션이 있다면 청산 시도
                self.exchange.create_order(
                    symbol=self.exchange.symbol,
                    side='SELL' if position['side'] == 'long' else 'BUY',
                    quantity=position['size'],
                    order_type='MARKET'
                )
            
            # 전략 상태 초기화
            self.reset()
            
        except Exception as e:
            self.logger.error(f"오류 복구 중 추가 오류 발생: {e}")

    def health_check(self) -> None:
        """시스템 헬스 체크"""
        try:
            # 포지션 상태 확인
            side, entry_price, leverage, amount = self.exchange.get_position()
            
            # 잔고 확인
            balance = self.exchange.get_balance()
            
            # RSI 확인
            ohlcv = self.exchange.get_ohlcv('1m', 14)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            rsi = RSIIndicator(df['close']).rsi()
            
            # 상태 출력
            status = {
                'position': side,
                'entry_price': entry_price,
                'leverage': leverage,
                'amount': amount,
                'balance': balance['free'],
                'rsi': rsi.iloc[-1]
            }
            
            logger.info(f"시스템 상태: {status}")
            
        except Exception as e:
            logger.error(f"헬스 체크 실패: {e}")

    def optimize_parameters(self) -> Dict[str, float]:
        """파라미터 최적화"""
        try:
            # 최적화 주기 체크 (매주 월요일)
            now = datetime.now()
            if self.last_optimization and (now - self.last_optimization).days < 7:
                return
            
            # 파라미터 그리드
            param_grid = {
                'rsi_long': [20, 25, 30],
                'rsi_short': [70, 75, 80],
                'stop_loss': [0.01, 0.02, 0.03],
                'take_profit': [0.02, 0.03, 0.04]
            }
            
            best_sharpe = -np.inf
            best_params = {}
            
            # 그리드 탐색
            for params in itertools.product(*param_grid.values()):
                sharpe = self.run_backtest(dict(zip(param_grid.keys(), params)))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = dict(zip(param_grid.keys(), params))
            
            # 설정 업데이트
            settings.RSI_LONG = best_params['rsi_long']
            settings.RSI_SHORT = best_params['rsi_short']
            settings.STOP_LOSS_PCT = best_params['stop_loss']
            settings.TAKE_PROFIT_PCT = best_params['take_profit']
            
            self.last_optimization = now
            logger.info(f"파라미터 최적화 완료: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"파라미터 최적화 실패: {e}")
            return {}

    def _analyze_trend(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """추세 분석"""
        try:
            # ADX 기반 추세 강도
            adx = indicators.get('adx', 25)
            trend_strength = 'strong' if adx >= 25 else 'weak'
            trend_direction = 'up' if indicators.get('macd', {}).get('histogram', 0) > 0 else 'down'
            
            # MACD 기반 추세 확인
            macd = indicators.get('macd', {})
            macd_state = 'bullish' if isinstance(macd, dict) and macd.get('histogram', 0) > 0 else 'bearish'
            
            # 볼린저 밴드 기반 추세 확인
            bb = indicators.get('bollinger', {})
            bb_state = 'normal'
            bb_width = 0.0
            if isinstance(bb, dict):
                bb_state = bb.get('state', 'normal')
                bb_width = bb.get('width', 0.0)
            
            return {
                'strength': trend_strength,
                'direction': trend_direction,
                'macd_state': macd_state,
                'bb_state': bb_state,
                'bb_width': bb_width
            }
            
        except Exception as e:
            logger.error(f"추세 분석 중 오류 발생: {e}")
            return {
                'strength': 'weak',
                'direction': 'neutral',
                'macd_state': 'neutral',
                'bb_state': 'normal',
                'bb_width': 0.0
            }

    def calculate_position_size(self, current_price: float, atr: float) -> float:
        """포지션 사이즈 계산"""
        try:
            # 계좌 잔고 가져오기
            account_balance = self.exchange.get_balance()
            if account_balance is None:
                self.logger.error("계좌 잔고를 가져올 수 없습니다.")
                return 0.0

            # 리스크 금액 계산 (계좌 잔고의 0.1%로 감소)
            risk_amount = account_balance * 0.001  # 0.5% -> 0.1%로 감소
            
            # ATR 기반 포지션 사이즈 계산
            position_size = risk_amount / (atr * 2)  # ATR의 2배를 리스크로 설정
            
            # 최대 포지션 사이즈 계산 (계좌 잔고의 95% 사용)
            max_position_size = (account_balance * self.config.leverage * 0.95) / current_price
            
            # 최소 주문 수량 확인
            min_qty = self.exchange.get_min_qty()
            if min_qty is None:
                self.logger.error("최소 주문 수량을 가져올 수 없습니다.")
                return 0.0
            
            # 포지션 사이즈 제한
            position_size = min(position_size, max_position_size)
            position_size = max(position_size, min_qty)
            
            # 마진 부족 체크
            required_margin = (position_size * current_price) / self.config.leverage
            if required_margin > account_balance * 0.95:
                self.logger.warning(f"마진 부족: 필요 마진 {required_margin:.2f} USDT, 사용 가능 잔고 {account_balance:.2f} USDT")
                position_size = (account_balance * 0.95 * self.config.leverage) / current_price
            
            # 로깅
            self.logger.info(f"포지션 사이즈 계산:")
            self.logger.info(f"- 계좌 잔고: {account_balance:.2f} USDT")
            self.logger.info(f"- 리스크 금액: {risk_amount:.2f} USDT")
            self.logger.info(f"- 현재 ATR14: {atr:.2f}")
            self.logger.info(f"- 계산된 포지션 사이즈: {position_size:.6f}")
            self.logger.info(f"- 최대 포지션 사이즈: {max_position_size:.6f}")
            self.logger.info(f"- 최소 주문 수량: {min_qty:.6f}")
            self.logger.info(f"- 필요 마진: {required_margin:.2f} USDT")
            self.logger.info(f"- 최종 포지션 사이즈: {position_size:.6f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 사이즈 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _get_buy_reason(self, market_data: Dict[str, Any]) -> str:
        """매수 신호 사유"""
        indicators = market_data.get('indicators', {})
        reasons = []
        
        if indicators.get('rsi', 50) < 30:
            reasons.append("RSI 과매도")
        if indicators.get('macd', {}).get('state') == 'bullish':
            reasons.append("MACD 상승 전환")
        if indicators.get('bollinger', {}).get('state') == 'oversold':
            reasons.append("볼린저 밴드 하단 터치")
        
        return ", ".join(reasons) if reasons else "기술적 지표 조합"

    def _get_sell_reason(self, market_data: Dict[str, Any]) -> str:
        """매도 신호 사유"""
        indicators = market_data.get('indicators', {})
        reasons = []
        
        if indicators.get('rsi', 50) > 70:
            reasons.append("RSI 과매수")
        if indicators.get('macd', {}).get('state') == 'bearish':
            reasons.append("MACD 하락 전환")
        if indicators.get('bollinger', {}).get('state') == 'overbought':
            reasons.append("볼린저 밴드 상단 터치")
        
        return ", ".join(reasons) if reasons else "기술적 지표 조합"

    def update_trade_result(self, is_profit: bool):
        """거래 결과 업데이트"""
        if is_profit:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def reset(self):
        """전략 초기화"""
        self.trailing_stop_active = False
        self.trailing_stop_price = 0.0
        self.last_signal = None
        self.last_signal_time = None
        self.consecutive_losses = 0
        self._indicators_cache.clear()
        self._last_calculation_time = None 
        self.peak_profit = 0.0  # 최고 수익률 초기화
        self.first_target_hit = False  # 1차 청산 상태 초기화 