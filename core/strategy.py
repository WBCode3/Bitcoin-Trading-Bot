from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from config.settings import settings
from utils.logger import setup_logger
from .exchange import ExchangeInterface
from .risk_manager import RiskManager
from datetime import datetime
import itertools

logger = setup_logger(__name__)

class TradingStrategy:
    def __init__(self, exchange: ExchangeInterface, risk_manager: RiskManager):
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.trailing_stop_active = False
        self.trailing_stop_price = 0.0
        self.last_optimization = None
        self.health_check_counter = 0

    def calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
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
            
            return rsi, macd, bb_lower, bb_upper
            
        except Exception as e:
            logger.error(f"지표 계산 실패: {e}")
            return pd.Series(), pd.Series(), pd.Series(), pd.Series()

    def detect_crash(self, df1: pd.DataFrame, df5: pd.DataFrame) -> bool:
        """크래시 감지"""
        try:
            # 5분 전 대비 가격 변화
            price_change = (df1['close'].iloc[-1] - df1['close'].iloc[-5]) / df1['close'].iloc[-5]
            
            # RSI 변화
            rsi1 = RSIIndicator(df1['close']).rsi()
            rsi_change = abs(rsi1.iloc[-1] - rsi1.iloc[-5])
            
            # MACD 음전환
            macd1 = MACD(df1['close']).macd_diff()
            macd_signal = macd1.iloc[-1] < 0 and macd1.iloc[-2] > 0
            
            return (price_change <= -0.015 and rsi_change > 12 and macd_signal)
            
        except Exception as e:
            logger.error(f"크래시 감지 실패: {e}")
            return False

    def detect_spike(self, df1: pd.DataFrame) -> bool:
        """스파이크 감지"""
        try:
            # 2분 전 대비 가격 변화
            price_change = (df1['close'].iloc[-1] - df1['close'].iloc[-2]) / df1['close'].iloc[-2]
            
            # RSI
            rsi = RSIIndicator(df1['close']).rsi()
            
            # MACD 양전환
            macd = MACD(df1['close']).macd_diff()
            macd_signal = macd.iloc[-1] > 0 and macd.iloc[-2] < 0
            
            return (price_change >= 0.02 and rsi.iloc[-1] > 75 and macd_signal)
            
        except Exception as e:
            logger.error(f"스파이크 감지 실패: {e}")
            return False

    def check_entry_conditions(self, df1: pd.DataFrame, df5: pd.DataFrame) -> Optional[str]:
        """듀얼 타임프레임 진입 조건 체크"""
        try:
            rsi1, macd1, bb_l1, bb_h1 = self.calculate_indicators(df1)
            rsi5, macd5, bb_l5, bb_h5 = self.calculate_indicators(df5)
            current_price = df1['close'].iloc[-1]
            
            # 공격적 롱 진입 (RSI < 20)
            if rsi1.iloc[-1] < 20 and macd1.iloc[-1] > 0:
                return 'aggressive_long'
            
            # 일반 롱 진입
            if (rsi1.iloc[-1] < 25 and rsi5.iloc[-1] < 25 and
                current_price <= bb_l1.iloc[-1] and
                macd1.iloc[-1] > 0 and macd5.iloc[-1] > 0):
                return 'long'
                
            # 숏 진입
            if (rsi1.iloc[-1] > 75 and rsi5.iloc[-1] > 75 and
                current_price >= bb_h1.iloc[-1] and
                macd1.iloc[-1] < 0 and macd5.iloc[-1] < 0):
                return 'short'
                
            return None
            
        except Exception as e:
            logger.error(f"진입 조건 체크 실패: {e}")
            return None

    def check_stop_loss(self, pnl: float) -> float:
        """분할 손절 체크"""
        if pnl <= -0.06:
            return 1.0  # 100% 청산
        elif pnl <= -0.04:
            return 0.7  # 70% 청산
        elif pnl <= -0.02:
            return 0.3  # 30% 청산
        return 0.0

    def check_exit_conditions(self, df: pd.DataFrame, side: str, entry_price: float) -> Tuple[bool, float]:
        """청산 조건 체크"""
        try:
            current_price = df['close'].iloc[-1]
            rsi, macd, _, _ = self.calculate_indicators(df)
            
            # PnL 계산
            if side == 'long':
                pnl = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) / entry_price
            
            # 분할 손절 체크
            close_pct = self.check_stop_loss(pnl)
            if close_pct > 0:
                return True, close_pct
            
            # 익절 조건
            if side == 'long':
                if current_price >= entry_price * (1 + settings.TAKE_PROFIT_PCT):
                    return True, 0.5  # 50% 익절
            else:
                if current_price <= entry_price * (1 - settings.TAKE_PROFIT_PCT):
                    return True, 0.5  # 50% 익절
            
            # 트레일링 스탑
            if settings.TRAILING_STOP:
                if side == 'long':
                    if current_price > self.trailing_stop_price:
                        self.trailing_stop_price = current_price
                    elif current_price <= self.trailing_stop_price * (1 - settings.TRAILING_STOP_PCT):
                        return True, 1.0
                else:
                    if current_price < self.trailing_stop_price:
                        self.trailing_stop_price = current_price
                    elif current_price >= self.trailing_stop_price * (1 + settings.TRAILING_STOP_PCT):
                        return True, 1.0
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"청산 조건 체크 실패: {e}")
            return False, 0.0

    def execute_trade(self) -> None:
        """거래 실행"""
        try:
            # OHLCV 데이터 가져오기
            ohlcv1 = self.exchange.get_ohlcv('1m', 100)
            ohlcv5 = self.exchange.get_ohlcv('5m', 100)
            df1 = pd.DataFrame(ohlcv1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df5 = pd.DataFrame(ohlcv5, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 현재 포지션 확인
            side, entry_price, leverage, amount = self.exchange.get_position()
            
            # Crash/Spike 감지
            if side:
                if self.detect_crash(df1, df5):
                    logger.warning("크래시 감지: 포지션 청산 및 숏 진입")
                    self.exchange.create_order('sell' if side == 'long' else 'buy', amount)
                    short_size = self.risk_manager.calculate_position_size(df1['close'].iloc[-1])
                    self.exchange.create_order('sell', short_size)
                    return
                    
                if self.detect_spike(df1):
                    logger.warning("스파이크 감지: 포지션 청산")
                    self.exchange.create_order('sell' if side == 'long' else 'buy', amount)
                    return
            
            if not side:  # 포지션이 없는 경우
                # 진입 조건 체크
                new_side = self.check_entry_conditions(df1, df5)
                if new_side:
                    # 포지션 사이즈 계산
                    position_size = self.risk_manager.calculate_position_size(df1['close'].iloc[-1])
                    if new_side == 'aggressive_long':
                        position_size = self.exchange.get_balance()['free'] / df1['close'].iloc[-1]
                    
                    if position_size > 0:
                        # 주문 실행
                        if 'long' in new_side:
                            self.exchange.create_order('buy', position_size)
                        else:
                            self.exchange.create_order('sell', position_size)
                        logger.info(f"새로운 포지션 진입: {new_side} {position_size}")
            
            else:  # 포지션이 있는 경우
                # 청산 조건 체크
                should_exit, close_pct = self.check_exit_conditions(df1, side, entry_price)
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