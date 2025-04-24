from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from config.settings import settings
from utils.logger import setup_logger
from .exchange import Exchange
from .risk_manager import RiskManager
from datetime import datetime, timedelta
import itertools
import logging
from decimal import Decimal
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator

logger = logging.getLogger(__name__)

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
        self.max_consecutive_losses = 5  # 연속 손실 제한 증가
        self.position_size_multiplier = 1.5  # 포지션 크기 증가
        self.leverage = 30  # 기본 레버리지 증가
        self.max_leverage = 75  # 최대 레버리지 증가
        self.min_leverage = 10  # 최소 레버리지 증가
        self.liquidation_buffer = 0.03  # 청산가 버퍼 증가 (3%)
        self.position_mode = 'hedge'
        self.logger = logging.getLogger(__name__)

    def calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, float]]:
        """기술적 지표 계산"""
        try:
            # RSI
            rsi = RSIIndicator(df['close'], window=14).rsi()
            
            # 볼린저 밴드
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            bb_lower = bb.bollinger_lband()
            bb_upper = bb.bollinger_hband()
            
            # MACD
            macd = MACD(df['close']).macd_diff()
            
            # 스토캐스틱
            stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            stoch_k = stoch.stoch()
            stoch_d = stoch.stoch_signal()
            
            # ADX
            adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
            adx_value = adx.adx()
            
            # 피보나치 레벨 계산
            fib_levels = self._calculate_fibonacci_levels(df)
            
            # 선물 특화 지표
            funding_rate = df['funding_rate'].iloc[-1] if 'funding_rate' in df.columns else 0
            open_interest = df['open_interest'].iloc[-1] if 'open_interest' in df.columns else 0
            oi_change = df['open_interest'].pct_change().iloc[-1] if 'open_interest' in df.columns else 0
            
            return rsi, macd, bb_lower, bb_upper, stoch_k, stoch_d, adx_value, fib_levels, funding_rate, open_interest, oi_change
            
        except Exception as e:
            logger.error(f"지표 계산 실패: {e}")
            return pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), {}, 0, 0, 0

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

    def check_entry_conditions(self, market_data: pd.DataFrame, current_index: int) -> Tuple[bool, str]:
        """진입 조건 확인"""
        try:
            if current_index < 20:  # 충분한 데이터가 없는 경우
                return False, ""
                
            current_data = market_data.iloc[current_index]
            prev_data = market_data.iloc[current_index - 1]
            
            # RSI 조건
            rsi = current_data['rsi']
            
            # MACD 조건
            macd = current_data['macd']
            macd_signal = current_data['macd_signal']
            prev_macd = prev_data['macd']
            prev_macd_signal = prev_data['macd_signal']
            
            # 볼린저 밴드 조건
            bb_high = current_data['bb_high']
            bb_low = current_data['bb_low']
            current_price = current_data['close']
            
            # 스토캐스틱 조건
            stoch_k = current_data['stoch_k']
            stoch_d = current_data['stoch_d']
            
            # 변동성 조건
            atr = current_data['atr']
            volatility = (atr / current_price) * 100
            
            # 거래량 조건
            volume = current_data['volume']
            volume_sma = current_data['volume_sma']
            volume_ratio = volume / volume_sma if volume_sma > 0 else 0
            
            # MACD 크로스 확인
            macd_cross_up = prev_macd < prev_macd_signal and macd > macd_signal
            macd_cross_down = prev_macd > prev_macd_signal and macd < macd_signal
            
            # 롱 포지션 진입 조건 (OR 조건으로 변경)
            long_conditions = [
                rsi < 40,  # RSI 과매도
                current_price < bb_low,  # 볼린저 밴드 하단 터치
                macd_cross_up and stoch_k < 30,  # MACD 상승 크로스와 스토캐스틱
                macd > macd_signal and stoch_k < 35  # MACD 상승 추세와 스토캐스틱
            ]
            
            # 숏 포지션 진입 조건 (OR 조건으로 변경)
            short_conditions = [
                rsi > 60,  # RSI 과매수
                current_price > bb_high,  # 볼린저 밴드 상단 터치
                macd_cross_down and stoch_k > 70,  # MACD 하락 크로스와 스토캐스틱
                macd < macd_signal and stoch_k > 65  # MACD 하락 추세와 스토캐스틱
            ]
            
            # 변동성과 거래량 조건 완화
            if volatility < 30 and volume_ratio > 1.0:  # 변동성 30% 이하, 거래량 1.0배 이상
                if any(long_conditions):
                    return True, "LONG"
                if any(short_conditions):
                    return True, "SHORT"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"진입 조건 확인 중 오류 발생: {e}")
            return False, ""

    def _adjust_leverage(self, volatility: float, trend_strength: bool) -> None:
        """변동성과 추세 강도에 따른 레버리지 조정"""
        try:
            # 기본 레버리지 설정
            base_leverage = self.leverage
            
            # 변동성에 따른 조정
            if volatility > 0.03:  # 높은 변동성
                base_leverage *= 0.7  # 조정 완화
            elif volatility < 0.01:  # 낮은 변동성
                base_leverage *= 1.5  # 조정 강화
            
            # 추세 강도에 따른 조정
            if trend_strength:
                base_leverage *= 1.3  # 조정 강화
            else:
                base_leverage *= 0.8  # 조정 완화
            
            # 레버리지 제한
            self.leverage = max(min(base_leverage, self.max_leverage), self.min_leverage)
            
        except Exception as e:
            logger.error(f"레버리지 조정 실패: {e}")

    def check_stop_loss(self, pnl: float, volatility: float, side: str, entry_price: float, current_price: float) -> bool:
        """손절 조건 확인"""
        try:
            # 손절 기준 완화
            if side == 'long':
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct > 0.05:  # 손절 기준 5%로 완화
                    return True
            else:  # short
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct > 0.05:  # 손절 기준 5%로 완화
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"손절 조건 확인 중 오류: {e}")
            return False

    def check_take_profit(self, pnl: float, volatility: float, side: str, entry_price: float, current_price: float) -> bool:
        """익절 조건 확인"""
        try:
            # 익절 기준 완화
            if side == 'long':
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct > 0.15:  # 익절 기준 15%로 완화
                    return True
            else:  # short
                profit_pct = (entry_price - current_price) / entry_price
                if profit_pct > 0.15:  # 익절 기준 15%로 완화
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"익절 조건 확인 중 오류: {e}")
            return False

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

    def check_exit_conditions(self, market_data: Dict[str, Any], side: str, entry_price: float) -> Tuple[bool, float]:
        """청산 조건 체크"""
        try:
            indicators = market_data['indicators']
            current_price = market_data['current_price']
            
            # RSI 상태 확인
            rsi_state = indicators['rsi']['state']
            rsi_short = indicators['rsi']['short_term']
            
            # 볼린저 밴드 상태 확인
            bb_state = indicators['bollinger']['state']
            
            # MACD 상태 확인
            macd_state = indicators['macd']['state']
            macd_hist = indicators['macd']['histogram']
            
            # 스토캐스틱 상태 확인
            stoch_state = indicators['stochastic']['state']
            stoch_cross = indicators['stochastic']['cross']
            
            # ADX 상태 확인
            adx_strength = indicators['adx']['trend_strength']
            adx_direction = indicators['adx']['trend_direction']
            
            # 일목균형표 상태 확인
            ichimoku_cloud = indicators['ichimoku']['cloud_state']
            ichimoku_base = indicators['ichimoku']['conversion_base']
            
            # 거래량 상태 확인
            volume_ratio = indicators['volume']['ratio']
            volume_trend = indicators['volume']['trend']
            
            # 트렌드 상태 확인
            trend_direction = indicators['trend']['direction']
            
            # 변동성 확인
            volatility = indicators['volatility']
            
            # 손익률 계산
            pnl = (current_price - entry_price) / entry_price if side == 'long' else (entry_price - current_price) / entry_price
            
            # 손절 조건
            if pnl < -0.02:  # 2% 손절
                return True, 1.0
            
            # 트레일링 스탑
            if pnl > 0.01:  # 1% 이상 수익 시 트레일링 스탑 활성화
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.trailing_stop_price = current_price
                else:
                    if side == 'long' and current_price > self.trailing_stop_price:
                        self.trailing_stop_price = current_price
                    elif side == 'short' and current_price < self.trailing_stop_price:
                        self.trailing_stop_price = current_price
                    
                    # 트레일링 스탑 도달 시 청산
                    if (side == 'long' and current_price < self.trailing_stop_price * 0.995) or \
                       (side == 'short' and current_price > self.trailing_stop_price * 1.005):
                        return True, 1.0
            
            # 롱 포지션 청산 조건
            if side == 'long':
                if (rsi_state == 'overbought' and rsi_short > 70 and
                    macd_state == 'bearish' and macd_hist < 0 and
                    stoch_cross == 'bearish' and
                    adx_strength == 'weak' and
                    ichimoku_cloud == 'below_cloud' and
                    volume_ratio < 0.8 and volume_trend == 'decreasing' and
                    trend_direction == 'down'):
                    return True, 1.0
            
            # 숏 포지션 청산 조건
            if side == 'short':
                if (rsi_state == 'oversold' and rsi_short < 30 and
                    macd_state == 'bullish' and macd_hist > 0 and
                    stoch_cross == 'bullish' and
                    adx_strength == 'weak' and
                    ichimoku_cloud == 'above_cloud' and
                    volume_ratio < 0.8 and volume_trend == 'decreasing' and
                    trend_direction == 'up'):
                    return True, 1.0
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"청산 조건 확인 중 오류 발생: {e}")
            return False, 0.0

    def execute_trade(self, market_data: Dict[str, Any]) -> None:
        """거래 실행"""
        try:
            # OHLCV 데이터 가져오기
            ohlcv15 = self.exchange.get_ohlcv('15m', 100)  # 15분 데이터
            ohlcv1h = self.exchange.get_ohlcv('1h', 100)   # 1시간 데이터 (추세 확인용)
            df15 = pd.DataFrame(ohlcv15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df1h = pd.DataFrame(ohlcv1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 현재 포지션 확인
            side, entry_price, leverage, amount = self.exchange.get_position()
            
            # Crash/Spike 감지 (15분 기준으로 조정)
            if side:
                if self.detect_crash(df15, df1h):
                    logger.warning("크래시 감지: 포지션 청산 및 숏 진입")
                    self.exchange.create_order('sell' if side == 'long' else 'buy', amount)
                    short_size = self._calculate_position_size(market_data)
                    self.exchange.create_order('sell', short_size)
                    return
                    
                if self.detect_spike(df15):
                    logger.warning("스파이크 감지: 포지션 청산")
                    self.exchange.create_order('sell' if side == 'long' else 'buy', amount)
                    return
            
            if not side:  # 포지션이 없는 경우
                # 진입 조건 체크
                signal = self.check_entry_conditions(df15, len(df15) - 1)
                if signal[0]:
                    # 포지션 사이즈 계산
                    position_size = self._calculate_position_size(market_data)
                    self.exchange.create_order(signal[1], position_size)
                    logger.info(f"새로운 포지션 진입: {signal[1]} {position_size} - 사유: {signal[1]}")
            
            else:  # 포지션이 있는 경우
                # 청산 조건 체크
                should_exit, close_pct = self.check_exit_conditions(market_data, side, entry_price)
                if should_exit:
                    # 포지션 청산
                    close_amount = amount * close_pct
                    self.exchange.create_order('sell' if side == 'long' else 'buy', close_amount)
                    logger.info(f"포지션 {close_pct*100}% 청산: {side} {close_amount}")
                    if close_pct == 1.0:
                        self.trailing_stop_active = False
                        self.trailing_stop_price = 0.0
            
            # 헬스 체크
            self.health_check_counter += 1
            if self.health_check_counter >= 60:
                self.health_check()
                self.health_check_counter = 0
            
        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
            raise

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

    def _validate_market_condition(self, market_data: Dict[str, Any]) -> bool:
        """시장 데이터 유효성 검증"""
        try:
            if 'indicators' not in market_data:
                logger.warning("시장 데이터에 indicators가 없습니다.")
                return False
                
            indicators = market_data['indicators']
            required_keys = ['volatility', 'trend', 'volume', 'rsi', 
                           'bollinger', 'macd', 'ichimoku', 
                           'stochastic', 'adx']
            
            for key in required_keys:
                if key not in indicators or indicators[key] is None:
                    logger.warning(f"필수 시장 데이터 누락: {key}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"시장 데이터 검증 중 오류 발생: {e}")
            return False

    def _calculate_trend(self, data: Dict[str, Any]) -> str:
        """추세 계산"""
        try:
            # MACD 히스토그램으로 추세 판단
            macd_hist = data['indicators']['macd']['histogram']
            if macd_hist > 0:
                return 'up'
            elif macd_hist < 0:
                return 'down'
            else:
                return 'neutral'
        except Exception as e:
            self.logger.warning(f"추세 계산 중 오류: {str(e)}")
            return 'neutral'

    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """매매 신호 생성"""
        try:
            # 추세 데이터 추가
            trend = self._calculate_trend(data)
            data['trend'] = trend
            
            # 시장 상태 평가
            market_condition = self._evaluate_market_condition(data)
            
            # 매매 신호 생성
            if market_condition['buy_signal']:
                return {'type': 'buy', 'confidence': market_condition['confidence']}
            elif market_condition['sell_signal']:
                return {'type': 'sell', 'confidence': market_condition['confidence']}
            else:
                return {'type': 'hold', 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"매매 신호 생성 중 오류: {str(e)}")
            return {'type': 'hold', 'confidence': 0.0}

    def _analyze_trend(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """추세 분석"""
        try:
            # ADX 기반 추세 강도
            adx = indicators.get('adx', {})
            trend_strength = adx.get('trend_strength', 'weak')
            trend_direction = adx.get('trend_direction', 'neutral')
            
            # MACD 기반 추세 확인
            macd = indicators.get('macd', {})
            macd_state = macd.get('state', 'neutral')
            
            # 일목균형표 기반 추세 확인
            ichimoku = indicators.get('ichimoku', {})
            cloud_state = ichimoku.get('cloud_state', 'neutral')
            conversion_base = ichimoku.get('conversion_base', 'neutral')
            
            # 볼린저 밴드 기반 추세 확인
            bb = indicators.get('bollinger', {})
            bb_state = bb.get('state', 'normal')
            bb_width = bb.get('width', 0.0)
            
            return {
                'strength': trend_strength,
                'direction': trend_direction,
                'macd_state': macd_state,
                'cloud_state': cloud_state,
                'conversion_base': conversion_base,
                'bb_state': bb_state,
                'bb_width': bb_width
            }
            
        except Exception as e:
            logger.error(f"추세 분석 중 오류 발생: {e}")
            return None

    def _check_buy_conditions(self, indicators: Dict[str, Any], trend: Dict[str, Any]) -> bool:
        """매수 조건 확인"""
        try:
            # 기본 조건
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            volatility = indicators.get('volatility', 0)
            
            # 조건 완화
            if (rsi < 35 and  # RSI 과매도 기준 완화
                macd > macd_signal and
                stoch_k > stoch_d and
                volatility > 1.5):  # 변동성 기준 완화
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"매수 조건 확인 중 오류: {e}")
            return False

    def _check_sell_conditions(self, indicators: Dict[str, Any], trend: Dict[str, Any]) -> bool:
        """매도 조건 확인"""
        try:
            # 기본 조건
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            volatility = indicators.get('volatility', 0)
            
            # 조건 완화
            if (rsi > 65 and  # RSI 과매수 기준 완화
                macd < macd_signal and
                stoch_k < stoch_d and
                volatility > 1.5):  # 변동성 기준 완화
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"매도 조건 확인 중 오류: {e}")
            return False

    def _calculate_position_size(self, market_data: Dict[str, Any]) -> float:
        """포지션 크기 계산"""
        try:
            indicators = market_data['indicators']
            risk_level = market_data['risk_level']
            
            # 기본 포지션 크기 (계좌의 30%)
            base_size = 0.3
            
            # 변동성 기반 조정
            volatility = indicators['volatility']
            if volatility > 0.05:  # 높은 변동성
                base_size *= 1.5  # 더 큰 포지션
            elif volatility < 0.02:  # 낮은 변동성
                base_size *= 0.7  # 더 작은 포지션
            
            # 거래량 기반 조정
            volume_ratio = indicators['volume']['ratio']
            if volume_ratio > 2.0:  # 매우 높은 거래량
                base_size *= 1.3
            elif volume_ratio < 1.0:  # 낮은 거래량
                base_size *= 0.5
            
            # 최소/최대 제한
            return max(min(base_size, 0.5), 0.1)  # 최소 10%, 최대 50%
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 중 오류 발생: {e}")
            return 0.3  # 기본값 반환

    def _get_buy_reason(self, market_data: Dict[str, Any]) -> str:
        """매수 신호 사유"""
        try:
            indicators = market_data['indicators']
            reasons = []
            
            if indicators['rsi']['state'] == 'oversold':
                reasons.append("RSI 과매도")
            if indicators['bollinger']['state'] == 'oversold':
                reasons.append("볼린저 밴드 과매도")
            if indicators['macd']['state'] == 'bullish':
                reasons.append("MACD 상승")
            if indicators['stochastic']['state'] == 'oversold':
                reasons.append("스토캐스틱 과매도")
                
            return ", ".join(reasons) if reasons else "기타 매수 조건 충족"
            
        except Exception as e:
            logger.error(f"매수 사유 생성 중 오류 발생: {e}")
            return "매수 조건 충족"

    def _get_sell_reason(self, market_data: Dict[str, Any]) -> str:
        """매도 신호 사유"""
        try:
            reasons = []
            
            if market_data['rsi']['state'] == 'overbought':
                reasons.append("RSI 과매수")
            if market_data['bollinger_bands']['state'] == 'overbought':
                reasons.append("볼린저 밴드 과매수")
            if market_data['macd']['state'] == 'bearish':
                reasons.append("MACD 하락")
            if market_data['stochastic']['state'] == 'overbought':
                reasons.append("스토캐스틱 과매수")
                
            return ", ".join(reasons) if reasons else "기타 매도 조건 충족"
            
        except Exception as e:
            logger.error(f"매도 사유 생성 중 오류 발생: {e}")
            return "매도 조건 충족"

    def update_trade_result(self, is_profit: bool):
        """거래 결과 업데이트"""
        try:
            if is_profit:
                self.consecutive_losses = 0
                self.position_size_multiplier = min(1.2, self.position_size_multiplier * 1.1)
            else:
                self.consecutive_losses += 1
                self.position_size_multiplier = max(0.5, self.position_size_multiplier * 0.9)
                
        except Exception as e:
            logger.error(f"거래 결과 업데이트 중 오류 발생: {e}")

    def reset(self):
        """전략 초기화"""
        try:
            self.last_signal = None
            self.last_signal_time = None
            self.consecutive_losses = 0
            self.position_size_multiplier = 1.0
            
        except Exception as e:
            logger.error(f"전략 초기화 중 오류 발생: {e}")

    def _evaluate_market_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """시장 상태 평가"""
        try:
            indicators = data['indicators']
            current_price = data['current_price']
            
            # 기본 설정
            result = {
                'buy_signal': False,
                'sell_signal': False,
                'confidence': 0.0
            }
            
            # RSI 상태 확인
            rsi = indicators.get('rsi', 50)
            rsi_state = 'oversold' if rsi < 35 else 'overbought' if rsi > 65 else 'neutral'  # RSI 기준 완화
            
            # 스토캐스틱 상태 확인
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            stoch_state = 'oversold' if stoch_k < 25 else 'overbought' if stoch_k > 75 else 'neutral'  # 스토캐스틱 기준 완화
            
            # 볼린저 밴드 상태 확인
            bb = indicators.get('bollinger', {})
            bb_upper = bb.get('upper', current_price * 1.02)
            bb_lower = bb.get('lower', current_price * 0.98)
            bb_state = 'overbought' if current_price > bb_upper else 'oversold' if current_price < bb_lower else 'normal'
            
            # MACD 상태 확인
            macd = indicators.get('macd', {})
            macd_hist = macd.get('histogram', 0)
            macd_state = 'bullish' if macd_hist > -0.05 else 'bearish'  # MACD 기준 완화
            
            # 거래량 상태 확인
            volume = indicators.get('volume', {})
            volume_ratio = volume.get('ratio', 1.0)
            
            # 변동성 확인
            volatility = indicators.get('volatility', 0.02)
            
            # 매수 조건
            if (rsi_state == 'oversold' and 
                stoch_state == 'oversold' and 
                bb_state == 'oversold' and 
                macd_state == 'bullish' and 
                volume_ratio > 1.1 and  # 거래량 기준 완화
                volatility < 0.05):  # 변동성 기준 완화
                result['buy_signal'] = True
                result['confidence'] = 0.8 if rsi < 30 and stoch_k < 20 else 0.6
            
            # 매도 조건
            elif (rsi_state == 'overbought' and 
                  stoch_state == 'overbought' and 
                  bb_state == 'overbought' and 
                  macd_state == 'bearish' and 
                  volume_ratio > 1.1 and  # 거래량 기준 완화
                  volatility < 0.05):  # 변동성 기준 완화
                result['sell_signal'] = True
                result['confidence'] = 0.8 if rsi > 70 and stoch_k > 80 else 0.6
            
            return result
            
        except Exception as e:
            self.logger.error(f"시장 상태 평가 중 오류: {str(e)}")
            return {
                'buy_signal': False,
                'sell_signal': False,
                'confidence': 0.0
            } 