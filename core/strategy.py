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

    def check_entry_conditions(self, df15: pd.DataFrame, df5: pd.DataFrame, df1h: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """진입 조건 확인 (멀티 타임프레임)"""
        try:
            # 1시간봉 데이터
            current_price_1h = df1h['close'].iloc[-1]
            rsi_1h = df1h['rsi'].iloc[-1]
            macd_1h = df1h['macd'].iloc[-1]
            macd_signal_1h = df1h['macd_signal'].iloc[-1]
            macd_hist_1h = df1h['macd_hist'].iloc[-1]
            stoch_k_1h = df1h['stoch_k'].iloc[-1]
            stoch_d_1h = df1h['stoch_d'].iloc[-1]
            bb_high_1h = df1h['bb_high'].iloc[-1]
            bb_low_1h = df1h['bb_low'].iloc[-1]
            atr_1h = df1h['atr'].iloc[-1]
            atr_14_1h = df1h['atr'].iloc[-14:].mean()
            atr_ema_1h = df1h['atr'].iloc[-10:].ewm(span=10).mean().iloc[-1]
            volume_1h = df1h['volume'].iloc[-1]
            volume_ma_1h = df1h['volume'].iloc[-20:-1].mean()
            adx_1h = df1h['adx'].iloc[-1]
            
            # 15분봉 데이터
            current_price_15 = df15['close'].iloc[current_idx]
            rsi_15 = df15['rsi'].iloc[current_idx]
            macd_15 = df15['macd'].iloc[current_idx]
            macd_signal_15 = df15['macd_signal'].iloc[current_idx]
            macd_hist_15 = df15['macd_hist'].iloc[current_idx]
            stoch_k_15 = df15['stoch_k'].iloc[current_idx]
            stoch_d_15 = df15['stoch_d'].iloc[current_idx]
            bb_high_15 = df15['bb_high'].iloc[current_idx]
            bb_low_15 = df15['bb_low'].iloc[current_idx]
            atr_15 = df15['atr'].iloc[current_idx]
            atr_14_15 = df15['atr'].iloc[current_idx-14:current_idx].mean()
            atr_ema_15 = df15['atr'].iloc[current_idx-10:current_idx].ewm(span=10).mean().iloc[-1]
            volume_15 = df15['volume'].iloc[current_idx]
            volume_ma_15 = df15['volume'].iloc[current_idx-20:current_idx].mean()
            adx_15 = df15['adx'].iloc[current_idx]
            
            # 5분봉 데이터
            current_price_5 = df5['close'].iloc[-1]
            rsi_5 = df5['rsi'].iloc[-1]
            macd_5 = df5['macd'].iloc[-1]
            macd_signal_5 = df5['macd_signal'].iloc[-1]
            macd_hist_5 = df5['macd_hist'].iloc[-1]
            stoch_k_5 = df5['stoch_k'].iloc[-1]
            stoch_d_5 = df5['stoch_d'].iloc[-1]
            bb_high_5 = df5['bb_high'].iloc[-1]
            bb_low_5 = df5['bb_low'].iloc[-1]
            atr_5 = df5['atr'].iloc[-1]
            atr_14_5 = df5['atr'].iloc[-14:].mean()
            atr_ema_5 = df5['atr'].iloc[-10:].ewm(span=10).mean().iloc[-1]
            volume_5 = df5['volume'].iloc[-1]
            volume_ma_5 = df5['volume'].iloc[-20:-1].mean()
            adx_5 = df5['adx'].iloc[-1]
            
            # 시장 레짐 필터 (ADX 기반)
            # 1. 횡보장 필터 (ADX < 20)
            if adx_1h < 20 or adx_15 < 20 or adx_5 < 20:
                self.logger.info("횡보장 감지: 거래 금지")
                return False, ""
            
            # 2. 강추세 필터 (ADX ≥ 35)
            if adx_1h < 35 or adx_15 < 35 or adx_5 < 35:
                self.logger.info("강추세 미달: 거래 금지")
                return False, ""
            
            # MACD 히스토그램 3봉 확인 (1시간봉)
            macd_hist_prev_1h = df1h['macd_hist'].iloc[-2]
            macd_hist_prev2_1h = df1h['macd_hist'].iloc[-3]
            macd_hist_prev3_1h = df1h['macd_hist'].iloc[-4]
            
            # MACD 히스토그램 3봉 확인 (15분봉)
            macd_hist_prev_15 = df15['macd_hist'].iloc[current_idx-1]
            macd_hist_prev2_15 = df15['macd_hist'].iloc[current_idx-2]
            macd_hist_prev3_15 = df15['macd_hist'].iloc[current_idx-3]
            
            # MACD 히스토그램 3봉 확인 (5분봉)
            macd_hist_prev_5 = df5['macd_hist'].iloc[-2]
            macd_hist_prev2_5 = df5['macd_hist'].iloc[-3]
            macd_hist_prev3_5 = df5['macd_hist'].iloc[-4]
            
            # 스토캐스틱 연속 교차 확인 (1시간봉)
            stoch_k_prev_1h = df1h['stoch_k'].iloc[-2]
            stoch_d_prev_1h = df1h['stoch_d'].iloc[-2]
            stoch_k_prev2_1h = df1h['stoch_k'].iloc[-3]
            stoch_d_prev2_1h = df1h['stoch_d'].iloc[-3]
            stoch_k_prev3_1h = df1h['stoch_k'].iloc[-4]
            stoch_d_prev3_1h = df1h['stoch_d'].iloc[-4]
            
            # 스토캐스틱 연속 교차 확인 (15분봉)
            stoch_k_prev_15 = df15['stoch_k'].iloc[current_idx-1]
            stoch_d_prev_15 = df15['stoch_d'].iloc[current_idx-1]
            stoch_k_prev2_15 = df15['stoch_k'].iloc[current_idx-2]
            stoch_d_prev2_15 = df15['stoch_d'].iloc[current_idx-2]
            stoch_k_prev3_15 = df15['stoch_k'].iloc[current_idx-3]
            stoch_d_prev3_15 = df15['stoch_d'].iloc[current_idx-3]
            
            # 스토캐스틱 연속 교차 확인 (5분봉)
            stoch_k_prev_5 = df5['stoch_k'].iloc[-2]
            stoch_d_prev_5 = df5['stoch_d'].iloc[-2]
            stoch_k_prev2_5 = df5['stoch_k'].iloc[-3]
            stoch_d_prev2_5 = df5['stoch_d'].iloc[-3]
            stoch_k_prev3_5 = df5['stoch_k'].iloc[-4]
            stoch_d_prev3_5 = df5['stoch_d'].iloc[-4]
            
            # 롱 포지션 진입 조건 (1시간봉)
            long_conditions_1h = [
                rsi_1h < 25,  # RSI 기준 강화
                macd_hist_1h > 0 and macd_hist_prev_1h > 0 and macd_hist_prev2_1h > 0 and macd_hist_prev3_1h > 0,  # 4봉 연속 양수
                stoch_k_1h > stoch_d_1h and stoch_k_prev_1h > stoch_d_prev_1h and stoch_k_prev2_1h > stoch_d_prev2_1h and stoch_k_prev3_1h > stoch_d_prev3_1h,  # 4봉 연속 교차
                atr_14_1h > atr_ema_1h * 1.3,  # ATR 필터 변경
                current_price_1h < bb_low_1h and (bb_low_1h - current_price_1h) / bb_low_1h > 0.01,  # 볼린저 밴드 이탈 기준 강화
                volume_1h > volume_ma_1h * 1.5  # 거래량 기준 강화
            ]
            
            # 롱 포지션 진입 조건 (15분봉)
            long_conditions_15 = [
                rsi_15 < 25,  # RSI 기준 강화
                macd_hist_15 > 0 and macd_hist_prev_15 > 0 and macd_hist_prev2_15 > 0 and macd_hist_prev3_15 > 0,  # 4봉 연속 양수
                stoch_k_15 > stoch_d_15 and stoch_k_prev_15 > stoch_d_prev_15 and stoch_k_prev2_15 > stoch_d_prev2_15 and stoch_k_prev3_15 > stoch_d_prev3_15,  # 4봉 연속 교차
                atr_14_15 > atr_ema_15 * 1.3,  # ATR 필터 변경
                current_price_15 < bb_low_15 and (bb_low_15 - current_price_15) / bb_low_15 > 0.01,  # 볼린저 밴드 이탈 기준 강화
                volume_15 > volume_ma_15 * 1.5  # 거래량 기준 강화
            ]
            
            # 롱 포지션 진입 조건 (5분봉)
            long_conditions_5 = [
                rsi_5 < 25,  # RSI 기준 강화
                macd_hist_5 > 0 and macd_hist_prev_5 > 0 and macd_hist_prev2_5 > 0 and macd_hist_prev3_5 > 0,  # 4봉 연속 양수
                stoch_k_5 > stoch_d_5 and stoch_k_prev_5 > stoch_d_prev_5 and stoch_k_prev2_5 > stoch_d_prev2_5 and stoch_k_prev3_5 > stoch_d_prev3_5,  # 4봉 연속 교차
                atr_14_5 > atr_ema_5 * 1.3,  # ATR 필터 변경
                current_price_5 < bb_low_5 and (bb_low_5 - current_price_5) / bb_low_5 > 0.01,  # 볼린저 밴드 이탈 기준 강화
                volume_5 > volume_ma_5 * 1.5  # 거래량 기준 강화
            ]
            
            # 숏 포지션 진입 조건 (1시간봉)
            short_conditions_1h = [
                rsi_1h > 75,  # RSI 기준 강화
                macd_hist_1h < 0 and macd_hist_prev_1h < 0 and macd_hist_prev2_1h < 0 and macd_hist_prev3_1h < 0,  # 4봉 연속 음수
                stoch_k_1h < stoch_d_1h and stoch_k_prev_1h < stoch_d_prev_1h and stoch_k_prev2_1h < stoch_d_prev2_1h and stoch_k_prev3_1h < stoch_d_prev3_1h,  # 4봉 연속 교차
                atr_14_1h > atr_ema_1h * 1.3,  # ATR 필터 변경
                current_price_1h > bb_high_1h and (current_price_1h - bb_high_1h) / bb_high_1h > 0.01,  # 볼린저 밴드 이탈 기준 강화
                volume_1h > volume_ma_1h * 1.5  # 거래량 기준 강화
            ]
            
            # 숏 포지션 진입 조건 (15분봉)
            short_conditions_15 = [
                rsi_15 > 75,  # RSI 기준 강화
                macd_hist_15 < 0 and macd_hist_prev_15 < 0 and macd_hist_prev2_15 < 0 and macd_hist_prev3_15 < 0,  # 4봉 연속 음수
                stoch_k_15 < stoch_d_15 and stoch_k_prev_15 < stoch_d_prev_15 and stoch_k_prev2_15 < stoch_d_prev2_15 and stoch_k_prev3_15 < stoch_d_prev3_15,  # 4봉 연속 교차
                atr_14_15 > atr_ema_15 * 1.3,  # ATR 필터 변경
                current_price_15 > bb_high_15 and (current_price_15 - bb_high_15) / bb_high_15 > 0.01,  # 볼린저 밴드 이탈 기준 강화
                volume_15 > volume_ma_15 * 1.5  # 거래량 기준 강화
            ]
            
            # 숏 포지션 진입 조건 (5분봉)
            short_conditions_5 = [
                rsi_5 > 75,  # RSI 기준 강화
                macd_hist_5 < 0 and macd_hist_prev_5 < 0 and macd_hist_prev2_5 < 0 and macd_hist_prev3_5 < 0,  # 4봉 연속 음수
                stoch_k_5 < stoch_d_5 and stoch_k_prev_5 < stoch_d_prev_5 and stoch_k_prev2_5 < stoch_d_prev2_5 and stoch_k_prev3_5 < stoch_d_prev3_5,  # 4봉 연속 교차
                atr_14_5 > atr_ema_5 * 1.3,  # ATR 필터 변경
                current_price_5 > bb_high_5 and (current_price_5 - bb_high_5) / bb_high_5 > 0.01,  # 볼린저 밴드 이탈 기준 강화
                volume_5 > volume_ma_5 * 1.5  # 거래량 기준 강화
            ]
            
            # 롱 진입 (세 타임프레임 모두 조건 충족)
            if all(long_conditions_1h) and all(long_conditions_15) and all(long_conditions_5):
                self.logger.info("롱 진입 조건 충족 (1시간, 15분, 5분)")
                return True, "LONG"
            
            # 숏 진입 (세 타임프레임 모두 조건 충족)
            if all(short_conditions_1h) and all(short_conditions_15) and all(short_conditions_5):
                self.logger.info("숏 진입 조건 충족 (1시간, 15분, 5분)")
                return True, "SHORT"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"진입 조건 확인 중 오류 발생: {e}")
            return False, ""

    def _adjust_leverage(self, market_data: Dict[str, Any]) -> None:
        """ATR 기반 레버리지 조정"""
        try:
            # ATR 데이터 수집 (14일)
            atr_series = market_data['indicators']['atr_series']  # ATR(14) 시계열
            current_atr = atr_series.iloc[-1]
            
            # ATR 백분위 계산
            atr_percentile = pd.qcut(atr_series, q=4, labels=False).iloc[-1]  # 0-3 범위의 백분위 (25%씩)
            
            # 변동성 구간별 레버리지 설정
            if atr_percentile == 3:  # 상위 25%
                self.leverage = 10
                leverage_desc = "상위 25%"
            elif atr_percentile == 2 or atr_percentile == 1:  # 중간 50%
                self.leverage = 30
                leverage_desc = "중간 50%"
            else:  # 하위 25%
                self.leverage = 50
                leverage_desc = "하위 25%"
            
            # 상세 로깅 추가
            self.logger.info(f"=== 레버리지 조정 정보 ===")
            self.logger.info(f"현재 ATR14: {current_atr:.4f}")
            self.logger.info(f"ATR 백분위: {atr_percentile} ({leverage_desc})")
            self.logger.info(f"적용 레버리지: {self.leverage}x")
            self.logger.info(f"=====================")
            
        except Exception as e:
            self.logger.error(f"레버리지 조정 중 오류 발생: {e}")
            # 오류 발생 시 안전한 기본값 사용
            self.leverage = 10

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
            rsi = indicators['rsi']['short_term']
            
            # MACD 상태 확인
            macd_hist = indicators['macd']['histogram']
            macd_hist_prev = indicators['macd']['histogram_prev']
            
            # 스토캐스틱 상태 확인
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
            stoch_k_prev = indicators['stoch_k_prev']
            stoch_d_prev = indicators['stoch_d_prev']
            
            # 손익률 계산
            pnl = (current_price - entry_price) / entry_price if side == 'long' else (entry_price - current_price) / entry_price
            
            # 1. 손절 조건 (단계별 청산)
            if pnl <= -0.03:  # -3.0% 달성 시 잔여량 전량 청산
                self.logger.info(f"2단계 손절 조건 충족: PNL {pnl:.2%}")
                return True, 1.0
            elif pnl <= -0.015:  # -1.5% 달성 시 30% 청산
                if not hasattr(self, 'first_stop_loss_hit'):
                    self.logger.info(f"1단계 손절 조건 충족: PNL {pnl:.2%}")
                    self.first_stop_loss_hit = True
                    return True, 0.3
            
            # 2. 익절 조건 (단계별 청산)
            if pnl >= 0.06:  # 6.0% 달성 시 잔여량 전량 청산
                self.logger.info(f"2단계 익절 조건 충족: PNL {pnl:.2%}")
                return True, 1.0
            elif pnl >= 0.02:  # 2.0% 달성 시 40% 청산
                if not hasattr(self, 'first_target_hit'):
                    self.logger.info(f"1단계 익절 조건 충족: PNL {pnl:.2%}")
                    self.first_target_hit = True
                    return True, 0.4
            
            # 3. 트레일링 스탑
            if pnl >= 0.02:  # 2.0% 이상 수익 시 트레일링 스탑 활성화
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.trailing_stop_price = current_price
                    self.logger.info("트레일링 스탑 활성화")
                else:
                    if side == 'long' and current_price > self.trailing_stop_price:
                        self.trailing_stop_price = current_price
                    elif side == 'short' and current_price < self.trailing_stop_price:
                        self.trailing_stop_price = current_price
                    
                    # 트레일링 스탑 도달 시 청산 (1.5% 하락)
                    if (side == 'long' and current_price < self.trailing_stop_price * 0.985) or \
                       (side == 'short' and current_price > self.trailing_stop_price * 1.015):
                        self.logger.info("트레일링 스탑 조건 충족")
                        return True, 1.0
            
            # 4. 반대 신호 청산
            # RSI 역전
            if (side == 'long' and rsi > 75) or (side == 'short' and rsi < 25):
                self.logger.info(f"RSI 역전 조건 충족: RSI {rsi}")
                return True, 1.0
            
            # MACD 히스토그램 반전
            if (side == 'long' and macd_hist < 0 and macd_hist_prev > 0) or \
               (side == 'short' and macd_hist > 0 and macd_hist_prev < 0):
                self.logger.info("MACD 히스토그램 반전 조건 충족")
                return True, 1.0
            
            # 스토캐스틱 %K/%D 역교차
            if (side == 'long' and stoch_k < stoch_d and stoch_k_prev > stoch_d_prev) or \
               (side == 'short' and stoch_k > stoch_d and stoch_k_prev < stoch_d_prev):
                self.logger.info("스토캐스틱 %K/%D 역교차 조건 충족")
                return True, 1.0
            
            return False, 0.0
            
        except Exception as e:
            self.logger.error(f"청산 조건 확인 중 오류 발생: {e}")
            return False, 0.0

    def execute_trade(self, market_data: Dict[str, Any]) -> None:
        """거래 실행 (지정가 주문 우선)"""
        try:
            # OHLCV 데이터 가져오기
            ohlcv15 = self.exchange.get_ohlcv('15m', 100)  # 15분 데이터
            ohlcv1h = self.exchange.get_ohlcv('1h', 100)   # 1시간 데이터 (추세 확인용)
            df15 = pd.DataFrame(ohlcv15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df1h = pd.DataFrame(ohlcv1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 현재 포지션 확인
            side, entry_price, leverage, amount = self.exchange.get_position()
            
            # ADX(14) 확인
            adx_14 = market_data['indicators']['adx_14']
            
            # 현재 가격과 슬리피지 계산
            current_price = market_data['current_price']
            slippage = current_price * 0.0002  # ±0.02%
            
            # 비상 시장가 주문 조건 (ADX ≥ 60)
            emergency_market_order = adx_14 >= 60
            
            if side:  # 포지션이 있는 경우
                # 청산 조건 체크
                should_exit, close_pct = self.check_exit_conditions(market_data, side, entry_price)
                if should_exit:
                    # 청산 수량 계산
                    close_amount = amount * close_pct
                    
                    # 비상 시장가 주문
                    if emergency_market_order:
                        self.logger.warning(f"비상 시장가 주문 실행 (ADX: {adx_14:.2f})")
                        self.exchange.create_order('sell' if side == 'long' else 'buy', close_amount, order_type='market')
                    else:
                        # 지정가 주문
                        if side == 'long':
                            limit_price = current_price + slippage  # 매도 지정가
                            self.exchange.create_order('sell', close_amount, order_type='limit', price=limit_price)
                            self.logger.info(f"롱 청산 지정가 주문: {close_amount} @ {limit_price}")
                        else:
                            limit_price = current_price - slippage  # 매수 지정가
                            self.exchange.create_order('buy', close_amount, order_type='limit', price=limit_price)
                            self.logger.info(f"숏 청산 지정가 주문: {close_amount} @ {limit_price}")
                    
                    if close_pct == 1.0:
                        self.trailing_stop_active = False
                        self.trailing_stop_price = 0.0
            
            else:  # 포지션이 없는 경우
                # 진입 조건 체크
                signal = self.check_entry_conditions(df15, df1h, len(df15) - 1)
                if signal[0]:
                    # 포지션 사이즈 계산
                    position_size = self._calculate_position_size(market_data)
                    
                    # 비상 시장가 주문
                    if emergency_market_order:
                        self.logger.warning(f"비상 시장가 주문 실행 (ADX: {adx_14:.2f})")
                        if signal[1] == "LONG":
                            self.exchange.create_order('buy', position_size, order_type='market')
                        else:
                            self.exchange.create_order('sell', position_size, order_type='market')
                    else:
                        # 지정가 주문
                        if signal[1] == "LONG":
                            limit_price = current_price - slippage  # 매수 지정가
                            self.exchange.create_order('buy', position_size, order_type='limit', price=limit_price)
                            self.logger.info(f"롱 진입 지정가 주문: {position_size} @ {limit_price}")
                        else:
                            limit_price = current_price + slippage  # 매도 지정가
                            self.exchange.create_order('sell', position_size, order_type='limit', price=limit_price)
                            self.logger.info(f"숏 진입 지정가 주문: {position_size} @ {limit_price}")
            
            # 헬스 체크
            self.health_check_counter += 1
            if self.health_check_counter >= 60:
                self.health_check()
                self.health_check_counter = 0
            
        except Exception as e:
            self.logger.error(f"거래 실행 실패: {e}")
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
        """포지션 크기 계산 (ATR 기반)"""
        try:
            # 기본 설정
            account_balance = market_data['account_balance']
            current_price = market_data['current_price']
            atr_14 = market_data['indicators']['atr_14']  # ATR(14) 값
            
            # 리스크 계산 (계좌 자산의 2%)
            risk_amount = account_balance * 0.02
            
            # 연속 손실 페널티 적용 (3연패 시 리스크 50% 축소)
            if self.consecutive_losses >= 3:
                risk_amount *= 0.5
                self.logger.info(f"연속 손실 페널티 적용: 리스크 {risk_amount:.2f}")
            
            # ATR 기반 포지션 사이즈 계산
            position_size = risk_amount / atr_14 if atr_14 > 0 else 0
            
            # 최소 주문 크기 확인
            min_order_size = market_data.get('min_order_size', 0.001)
            position_size = max(position_size, min_order_size)
            
            # 최대 주문 크기 제한 (레버리지 고려)
            max_position_size = account_balance * self.leverage * 0.95  # 95%로 제한
            position_size = min(position_size, max_position_size)
            
            # 상세 로깅 추가
            self.logger.info(f"=== 포지션 사이즈 계산 정보 ===")
            self.logger.info(f"계좌 자산: {account_balance:.2f}")
            self.logger.info(f"기본 리스크 금액: {account_balance * 0.02:.2f}")
            self.logger.info(f"적용 리스크 금액: {risk_amount:.2f}")
            self.logger.info(f"현재 ATR14: {atr_14:.4f}")
            self.logger.info(f"계산된 포지션 크기: {position_size:.4f}")
            self.logger.info(f"적용 레버리지: {self.leverage}x")
            self.logger.info(f"최대 가능 포지션 크기: {max_position_size:.4f}")
            self.logger.info(f"=====================")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 중 오류 발생: {e}")
            return 0.0

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
            stoch_state = 'oversold' if stoch_k < 30 else 'overbought' if stoch_k > 70 else 'neutral'  # 스토캐스틱 범위 30/70으로 변경
            
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
                result['confidence'] = 0.8 if rsi < 30 and stoch_k < 30 else 0.6
            
            # 매도 조건
            elif (rsi_state == 'overbought' and 
                  stoch_state == 'overbought' and 
                  bb_state == 'overbought' and 
                  macd_state == 'bearish' and 
                  volume_ratio > 1.1 and  # 거래량 기준 완화
                  volatility < 0.05):  # 변동성 기준 완화
                result['sell_signal'] = True
                result['confidence'] = 0.8 if rsi > 70 and stoch_k > 70 else 0.6
            
            return result
            
        except Exception as e:
            self.logger.error(f"시장 상태 평가 중 오류: {str(e)}")
            return {
                'buy_signal': False,
                'sell_signal': False,
                'confidence': 0.0
            } 