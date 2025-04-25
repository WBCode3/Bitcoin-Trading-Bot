import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.volume import VolumeWeightedAveragePrice
from config.settings import settings
from utils.logger import setup_logger
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketConfig:
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: int = 2
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    vwap_period: int = 20
    volatility_threshold: float = 0.02
    volume_threshold: float = 1.5

class MarketAnalyzer:
    def __init__(self, config: MarketConfig = None):
        self.config = config or MarketConfig()
        self.logger = logging.getLogger(__name__)
        self._indicators_cache = {}
        self._last_calculation_time = None
        self.volatility_thresholds = {
            'very_low': 0.005,
            'low': 0.01,
            'medium': 0.02,
            'high': 0.03,
            'very_high': 0.05
        }
        self.rsi_periods = {
            'short': 14,  # 기본 RSI 기간
            'long': 28    # 장기 RSI 기간
        }
        self.rsi_thresholds = {
            'overbought': 70,
            'oversold': 30
        }
        self.bb_periods = {
            'short': 20,
            'long': 50
        }
        self.bb_std = {
            'short': 2.0,
            'long': 2.5
        }
        self.macd_params = {
            'fast': 12,
            'slow': 26,
            'signal': 9
        }
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.stoch_params = {'k': 14, 'd': 3, 'smooth': 3}
        self.adx_period = 14
        self.min_data_points = 100  # 최소 필요한 데이터 포인트 수

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검증"""
        try:
            # 최소 데이터 포인트 확인
            if len(df) < self.min_data_points:
                return False
                
            # 필수 컬럼 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                return False
                
            # NaN 값 확인
            if df[required_columns].isnull().any().any():
                return False
                
            # 데이터 타입 검증
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False
                    
            return True
            
        except Exception:
            return False

    def analyze_market_condition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시장 상태 분석 (캐싱 적용)"""
        try:
            current_time = datetime.now()
            cache_key = f"{df.index[-1]}_{len(df)}"
            
            # 캐시된 결과가 있고 5분 이내라면 재사용
            if (cache_key in self._indicators_cache and 
                self._last_calculation_time and 
                (current_time - self._last_calculation_time) < timedelta(minutes=5)):
                return self._indicators_cache[cache_key]

            analysis = self._analyze_all_indicators(df)
            self._indicators_cache[cache_key] = analysis
            self._last_calculation_time = current_time
            return analysis
            
        except Exception as e:
            self.logger.error(f"시장 분석 실패: {e}")
            return self._get_default_analysis()

    def _analyze_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """모든 기술적 지표 분석"""
        # RSI
        rsi = RSIIndicator(df['close'], window=self.config.rsi_period).rsi().iloc[-1]
        
        # 볼린저 밴드
        bb = BollingerBands(df['close'], window=self.config.bb_period, window_dev=self.config.bb_std)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        
        # MACD
        macd_indicator = MACD(
            df['close'],
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal
        )
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_hist = macd_indicator.macd_diff().iloc[-1]
        
        # ADX
        adx = ADXIndicator(
            df['high'],
            df['low'],
            df['close'],
            window=self.config.adx_period
        )
        adx_value = adx.adx().iloc[-1]
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            window=self.config.vwap_period
        ).volume_weighted_average_price().iloc[-1]
        
        # 변동성
        volatility = self.calculate_volatility(df)
        
        # 거래량
        volume = df['volume'].iloc[-1]
        volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
        
        return {
            'rsi': {
                'value': rsi,
                'state': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'normal'
            },
            'bollinger': {
                'upper': bb_upper,
                'lower': bb_lower,
                'middle': bb_middle,
                'state': 'overbought' if df['close'].iloc[-1] > bb_upper else 'oversold' if df['close'].iloc[-1] < bb_lower else 'normal'
            },
            'macd': {
                'value': macd,
                'signal': macd_signal,
                'histogram': macd_hist,
                'state': 'bullish' if macd_hist > 0 else 'bearish'
            },
            'adx': {
                'value': adx_value,
                'state': 'strong' if adx_value > 25 else 'weak'
            },
            'vwap': vwap,
            'volatility': volatility,
            'volume': {
                'current': volume,
                'ma': volume_ma,
                'ratio': volume_ratio,
                'state': 'high' if volume_ratio > self.config.volume_threshold else 'normal'
            }
        }

    def _get_default_analysis(self) -> Dict[str, Any]:
        """기본 분석 결과 반환"""
        return {
            'rsi': {'value': 50, 'state': 'normal'},
            'bollinger': {'upper': 0, 'lower': 0, 'middle': 0, 'state': 'normal'},
            'macd': {'value': 0, 'signal': 0, 'histogram': 0, 'state': 'neutral'},
            'adx': {'value': 25, 'state': 'weak'},
            'vwap': 0,
            'volatility': 0.02,
            'volume': {'current': 0, 'ma': 0, 'ratio': 1.0, 'state': 'normal'}
        }

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """변동성 계산 (ATR 기반)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # ATR 계산
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # 변동성 비율 계산
            volatility = atr / close.iloc[-1]
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"변동성 계산 실패: {e}")
            return self.config.volatility_threshold

    def check_market_health(self, analysis: Dict[str, Any]) -> bool:
        """시장 상태 건강성 검사"""
        try:
            # RSI 검사
            rsi = analysis['rsi']['value']
            if rsi < 20 or rsi > 80:  # 극단적인 과매수/과매도
                return False
            
            # 볼린저 밴드 검사
            bb_state = analysis['bollinger']['state']
            if bb_state == 'overbought' or bb_state == 'oversold':
                return False
            
            # ADX 검사
            adx = analysis['adx']['value']
            if adx > 50:  # 너무 강한 추세
                return False
            
            # 거래량 검사
            volume_ratio = analysis['volume']['ratio']
            if volume_ratio > 3.0:  # 비정상적인 거래량
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"시장 상태 검사 실패: {e}")
            return False

    def reset(self):
        """분석기 초기화"""
        self._indicators_cache.clear()
        self._last_calculation_time = None

    def analyze_trend(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """트렌드 분석"""
        try:
            if len(df) < 50:  # 최소 데이터 포인트 수정
                return None
                
            # 이동평균 계산
            ma_short = df['close'].rolling(window=20).mean()
            ma_medium = df['close'].rolling(window=50).mean()
            
            # 트렌드 강도 계산
            trend_strength = self._calculate_trend_strength(ma_short, ma_medium)
            
            return {
                'short_term': float(ma_short.iloc[-1]),
                'medium_term': float(ma_medium.iloc[-1]),
                'strength': float(trend_strength),
                'direction': 'up' if ma_short.iloc[-1] > ma_medium.iloc[-1] else 'down'
            }
            
        except Exception:
            return None

    def _calculate_trend_strength(self, ma_short: pd.Series, ma_medium: pd.Series) -> float:
        """트렌드 강도 계산"""
        try:
            if ma_medium.iloc[-1] == 0:
                return 0.5
                
            short_vs_medium = (ma_short.iloc[-1] - ma_medium.iloc[-1]) / ma_medium.iloc[-1]
            strength = abs(short_vs_medium)
            return float(min(max(strength, 0), 1))
            
        except Exception:
            return 0.5

    def analyze_volume(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """거래량 분석"""
        try:
            if len(df) < 20:  # 이동평균을 위한 최소 데이터 필요
                logger.warning("거래량 분석을 위한 데이터가 부족합니다.")
                return None
                
            # 거래량 이동평균
            volume_ma = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1]
            
            # VWAP 계산
            vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            return {
                'volume_ma': float(volume_ma.iloc[-1]),
                'volume_ratio': float(volume_ratio),
                'vwap': float(vwap.iloc[-1]),
                'volume_trend': 'increasing' if volume_ratio > 1.2 else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"거래량 분석 중 오류 발생: {e}")
            return None

    def analyze_rsi(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """RSI 분석"""
        try:
            if len(df) < self.rsi_periods['long']:
                logger.warning("RSI 분석을 위한 데이터가 부족합니다.")
                return None
                
            # 가격 변화 계산
            delta = df['close'].diff()
            
            # 단기 RSI
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_periods['short']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_periods['short']).mean()
            rs = gain / loss
            rsi_short = 100 - (100 / (1 + rs))
            
            # 장기 RSI
            gain_long = (delta.where(delta > 0, 0)).rolling(window=self.rsi_periods['long']).mean()
            loss_long = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_periods['long']).mean()
            rs_long = gain_long / loss_long
            rsi_long = 100 - (100 / (1 + rs_long))
            
            # RSI 상태 판단
            current_rsi_short = rsi_short.iloc[-1]
            current_rsi_long = rsi_long.iloc[-1]
            
            rsi_state = 'neutral'
            if current_rsi_short > self.rsi_thresholds['overbought']:
                rsi_state = 'overbought'
            elif current_rsi_short < self.rsi_thresholds['oversold']:
                rsi_state = 'oversold'
                
            # RSI 다이버전스 확인
            divergence = self._check_rsi_divergence(df, rsi_short)
            
            return {
                'short_term': float(current_rsi_short),
                'long_term': float(current_rsi_long),
                'state': rsi_state,
                'divergence': divergence,
                'trend': 'up' if current_rsi_short > current_rsi_long else 'down'
            }
            
        except Exception as e:
            logger.error(f"RSI 분석 중 오류 발생: {e}")
            return None

    def _check_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> str:
        """RSI 다이버전스 확인"""
        try:
            lookback = 20
            if len(df) < lookback:
                return 'none'
                
            # 상승 다이버전스
            if (df['low'].iloc[-1] < df['low'].iloc[-lookback] and 
                rsi.iloc[-1] > rsi.iloc[-lookback]):
                return 'bullish'
                
            # 하락 다이버전스
            if (df['high'].iloc[-1] > df['high'].iloc[-lookback] and 
                rsi.iloc[-1] < rsi.iloc[-lookback]):
                return 'bearish'
                
            return 'none'
            
        except Exception as e:
            logger.error(f"RSI 다이버전스 확인 중 오류 발생: {e}")
            return 'none'

    def analyze_bollinger_bands(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """볼린저 밴드 분석"""
        try:
            if len(df) < self.bb_periods['long']:
                logger.warning("볼린저 밴드 분석을 위한 데이터가 부족합니다.")
                return None
                
            # 단기 볼린저 밴드
            bb_short = self.calculate_bollinger_bands(df, self.bb_periods['short'], self.bb_std['short'])
            
            # 장기 볼린저 밴드
            bb_long = self.calculate_bollinger_bands(df, self.bb_periods['long'], self.bb_std['long'])
            
            current_price = df['close'].iloc[-1]
            bb_short_upper = bb_short['upper'].iloc[-1]
            bb_short_middle = bb_short['middle'].iloc[-1]
            bb_short_lower = bb_short['lower'].iloc[-1]
            
            # 볼린저 밴드 상태
            bb_state = 'normal'
            if current_price > bb_short_upper:
                bb_state = 'overbought'
            elif current_price < bb_short_lower:
                bb_state = 'oversold'
                
            # 볼린저 밴드 폭
            bb_width = (bb_short_upper - bb_short_lower) / bb_short_middle
            
            return {
                'short': {
                    'upper': float(bb_short_upper),
                    'middle': float(bb_short_middle),
                    'lower': float(bb_short_lower)
                },
                'long': {
                    'upper': float(bb_long['upper'].iloc[-1]),
                    'middle': float(bb_long['middle'].iloc[-1]),
                    'lower': float(bb_long['lower'].iloc[-1])
                },
                'state': bb_state,
                'width': float(bb_width),
                'squeeze': bb_width < 0.1
            }
            
        except Exception as e:
            logger.error(f"볼린저 밴드 분석 중 오류 발생: {e}")
            return None

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            if len(data) < period:
                logger.warning("볼린저 밴드 계산을 위한 데이터가 부족합니다.")
                return {
                    'middle': pd.Series([data['close'].iloc[-1]] * len(data), index=data.index),
                    'upper': pd.Series([data['close'].iloc[-1]] * len(data), index=data.index),
                    'lower': pd.Series([data['close'].iloc[-1]] * len(data), index=data.index)
                }

            # 종가 데이터 가져오기
            close = data['close'].copy()
            
            # NaN 값 처리
            if close.isnull().any():
                close = close.ffill().bfill()
                logger.warning("종가 데이터에 NaN 값이 존재하여 전진/후진 채움을 수행했습니다.")

            # 중간 밴드 (SMA) 계산
            middle = close.rolling(window=period, min_periods=1).mean()
            
            # 표준편차 계산
            std_dev = close.rolling(window=period, min_periods=1).std()
            
            # 상단/하단 밴드 계산
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            # NaN 값 처리
            middle = middle.ffill().bfill()
            upper = upper.ffill().bfill()
            lower = lower.ffill().bfill()
            
            return {
                'middle': middle,
                'upper': upper,
                'lower': lower
            }
            
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 중 오류 발생: {e}")
            # 오류 발생 시 기본값 반환
            return {
                'middle': pd.Series([data['close'].iloc[-1]] * len(data), index=data.index),
                'upper': pd.Series([data['close'].iloc[-1]] * len(data), index=data.index),
                'lower': pd.Series([data['close'].iloc[-1]] * len(data), index=data.index)
            }

    def analyze_macd(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """MACD 분석"""
        try:
            if len(df) < self.macd_params['slow']:
                logger.warning("MACD 분석을 위한 데이터가 부족합니다.")
                return None
                
            exp1 = df['close'].ewm(span=self.macd_params['fast'], adjust=False).mean()
            exp2 = df['close'].ewm(span=self.macd_params['slow'], adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.macd_params['signal'], adjust=False).mean()
            histogram = macd - signal
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_hist = histogram.iloc[-1]
            
            # MACD 상태
            macd_state = 'neutral'
            if current_macd > current_signal and current_hist > 0:
                macd_state = 'bullish'
            elif current_macd < current_signal and current_hist < 0:
                macd_state = 'bearish'
                
            # MACD 히스토그램 방향
            hist_direction = 'up' if current_hist > histogram.iloc[-2] else 'down'
            
            return {
                'macd': float(current_macd),
                'signal': float(current_signal),
                'histogram': float(current_hist),
                'state': macd_state,
                'hist_direction': hist_direction
            }
            
        except Exception as e:
            logger.error(f"MACD 분석 중 오류 발생: {e}")
            return None

    def analyze_fibonacci(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """피보나치 분석"""
        try:
            lookback = 100
            if len(df) < lookback:
                logger.warning("피보나치 분석을 위한 데이터가 부족합니다.")
                return None
                
            high = df['high'].rolling(window=lookback).max()
            low = df['low'].rolling(window=lookback).min()
            
            current_high = high.iloc[-1]
            current_low = low.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            diff = current_high - current_low
            if diff == 0:
                logger.warning("피보나치 레벨 계산을 위한 가격 범위가 0입니다.")
                return None
                
            levels = {
                '0': float(current_low),
                '0.236': float(current_low + diff * 0.236),
                '0.382': float(current_low + diff * 0.382),
                '0.5': float(current_low + diff * 0.5),
                '0.618': float(current_low + diff * 0.618),
                '0.786': float(current_low + diff * 0.786),
                '1': float(current_high)
            }
            
            current_level = None
            for level, price in levels.items():
                if current_price <= price:
                    current_level = level
                    break
                    
            return {
                'levels': levels,
                'current_level': current_level,
                'high': float(current_high),
                'low': float(current_low),
                'retracement': float((current_price - current_low) / diff)
            }
            
        except Exception as e:
            logger.error(f"피보나치 분석 중 오류 발생: {e}")
            return None

    def analyze_ichimoku(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """일목균형표 분석"""
        try:
            if len(df) < 52:  # 최장 기간 이동평균을 위한 최소 데이터 필요
                logger.warning("일목균형표 분석을 위한 데이터가 부족합니다.")
                return None
                
            conversion = (df['high'].rolling(window=9).max() + 
                        df['low'].rolling(window=9).min()) / 2
            
            base = (df['high'].rolling(window=26).max() + 
                   df['low'].rolling(window=26).min()) / 2
            
            leading_span1 = ((conversion + base) / 2).shift(26)
            leading_span2 = ((df['high'].rolling(window=52).max() + 
                            df['low'].rolling(window=52).min()) / 2).shift(26)
            
            lagging_span = df['close'].shift(-26)
            
            current_price = df['close'].iloc[-1]
            
            cloud_state = 'neutral'
            if current_price > leading_span1.iloc[-1] and current_price > leading_span2.iloc[-1]:
                cloud_state = 'above_cloud'
            elif current_price < leading_span1.iloc[-1] and current_price < leading_span2.iloc[-1]:
                cloud_state = 'below_cloud'
                
            conversion_base = 'bullish' if conversion.iloc[-1] > base.iloc[-1] else 'bearish'
            
            return {
                'conversion': float(conversion.iloc[-1]),
                'base': float(base.iloc[-1]),
                'leading_span1': float(leading_span1.iloc[-1]),
                'leading_span2': float(leading_span2.iloc[-1]),
                'lagging_span': float(lagging_span.iloc[-1]),
                'cloud_state': cloud_state,
                'conversion_base': conversion_base
            }
            
        except Exception as e:
            logger.error(f"일목균형표 분석 중 오류 발생: {e}")
            return None

    def analyze_stochastic(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """스토캐스틱 분석"""
        try:
            if len(df) < self.stoch_params['k']:
                logger.warning("스토캐스틱 분석을 위한 데이터가 부족합니다.")
                return None
                
            low_min = df['low'].rolling(window=self.stoch_params['k']).min()
            high_max = df['high'].rolling(window=self.stoch_params['k']).max()
            
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=self.stoch_params['d']).mean()
            
            current_k = k.iloc[-1]
            current_d = d.iloc[-1]
            
            stoch_state = 'neutral'
            if current_k > 80 and current_d > 80:
                stoch_state = 'overbought'
            elif current_k < 20 and current_d < 20:
                stoch_state = 'oversold'
                
            cross = 'none'
            if current_k > current_d and k.iloc[-2] <= d.iloc[-2]:
                cross = 'golden'
            elif current_k < current_d and k.iloc[-2] >= d.iloc[-2]:
                cross = 'dead'
                
            return {
                'k': float(current_k),
                'd': float(current_d),
                'state': stoch_state,
                'cross': cross
            }
            
        except Exception as e:
            logger.error(f"스토캐스틱 분석 중 오류 발생: {e}")
            return None

    def analyze_adx(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """ADX 분석"""
        try:
            if len(df) < self.adx_period:
                logger.warning("ADX 분석을 위한 데이터가 부족합니다.")
                return None
                
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.adx_period).mean()
            
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr14 = tr.rolling(window=self.adx_period).sum()
            plus_di14 = 100 * (plus_dm.rolling(window=self.adx_period).sum() / tr14)
            minus_di14 = 100 * (minus_dm.rolling(window=self.adx_period).sum() / tr14)
            
            dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
            adx = dx.rolling(window=self.adx_period).mean()
            
            current_adx = adx.iloc[-1]
            current_plus_di = plus_di14.iloc[-1]
            current_minus_di = minus_di14.iloc[-1]
            
            trend_strength = 'weak'
            if current_adx > 25:
                trend_strength = 'strong'
            elif current_adx > 20:
                trend_strength = 'moderate'
                
            trend_direction = 'up' if current_plus_di > current_minus_di else 'down'
            
            return {
                'adx': float(current_adx),
                'plus_di': float(current_plus_di),
                'minus_di': float(current_minus_di),
                'trend_strength': trend_strength,
                'trend_direction': trend_direction
            }
            
        except Exception as e:
            logger.error(f"ADX 분석 중 오류 발생: {e}")
            return None

    def _analyze_volatility(self, indicators: Dict[str, Any]) -> str:
        """변동성 분석"""
        try:
            volatility = indicators.get('volatility', 0.02)
            
            if volatility > self.volatility_thresholds['very_high']:
                return 'very_high'
            elif volatility > self.volatility_thresholds['high']:
                return 'high'
            elif volatility > self.volatility_thresholds['medium']:
                return 'medium'
            elif volatility > self.volatility_thresholds['low']:
                return 'low'
            else:
                return 'very_low'
                
        except Exception as e:
            self.logger.error(f"변동성 분석 중 오류 발생: {e}")
            return 'medium'

    def _analyze_trend(self, indicators: Dict[str, Any]) -> str:
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
            
            if trend_strength == 'strong' and trend_direction == 'up':
                return 'strong_uptrend'
            elif trend_strength == 'strong' and trend_direction == 'down':
                return 'strong_downtrend'
            elif macd_state == 'bullish' and cloud_state == 'above_cloud':
                return 'uptrend'
            elif macd_state == 'bearish' and cloud_state == 'below_cloud':
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"추세 분석 중 오류 발생: {e}")
            return 'sideways'

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            # 가격 변화 계산
            deltas = prices.diff()
            
            # 상승/하락 구분
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            # 평균 계산
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # RSI 계산
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"RSI 계산 중 오류 발생: {str(e)}")
            return 50.0  # 기본값

    def analyze_momentum(self, prices: pd.Series) -> Dict[str, Any]:
        """
        시장 모멘텀 분석
        """
        try:
            # RSI 계산
            rsi_value = self._calculate_rsi(prices)
            
            # MACD 계산
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # 스토캐스틱 계산
            low_min = prices.rolling(window=14).min()
            high_max = prices.rolling(window=14).max()
            k = 100 * (prices - low_min) / (high_max - low_min)
            d = k.rolling(window=3).mean()
            
            # 모멘텀 점수 계산 (0-100)
            momentum_score = 0
            
            # RSI 기반 점수 (0-40)
            if rsi_value > 70:
                momentum_score += 40
            elif rsi_value > 60:
                momentum_score += 30
            elif rsi_value > 50:
                momentum_score += 20
            elif rsi_value > 40:
                momentum_score += 10
            
            # MACD 기반 점수 (0-30)
            if histogram.iloc[-1] > 0:
                momentum_score += 30
            elif histogram.iloc[-1] > -0.5:
                momentum_score += 15
            
            # 스토캐스틱 기반 점수 (0-30)
            if k.iloc[-1] > 80:
                momentum_score += 30
            elif k.iloc[-1] > 60:
                momentum_score += 20
            elif k.iloc[-1] > 40:
                momentum_score += 10
            
            momentum_state = 'strong' if momentum_score > 80 else \
                            'moderate' if momentum_score > 50 else \
                            'weak'
            
            return {
                'score': momentum_score,
                'state': momentum_state,
                'rsi': rsi_value,
                'macd': {
                    'value': float(macd.iloc[-1]),
                    'signal': float(signal.iloc[-1]),
                    'histogram': float(histogram.iloc[-1])
                },
                'stochastic': {
                    'k': float(k.iloc[-1]),
                    'd': float(d.iloc[-1])
                }
            }
            
        except Exception as e:
            self.logger.error(f"모멘텀 분석 중 오류 발생: {str(e)}")
            return {
                'score': 50,
                'state': 'neutral',
                'rsi': 50,
                'macd': {'value': 0, 'signal': 0, 'histogram': 0},
                'stochastic': {'k': 50, 'd': 50}
            }

    def _evaluate_risk_level(self, volatility: str, trend: str, momentum: str) -> str:
        """리스크 레벨 평가"""
        try:
            # 변동성 기반 리스크 평가
            if volatility == 'very_high':
                return 'very_high'
            elif volatility == 'high':
                return 'high'
                
            # 추세 기반 리스크 평가
            if trend == 'strong_uptrend':
                return 'low'
            elif trend == 'strong_downtrend':
                return 'high'
                
            # 모멘텀 기반 리스크 평가
            if momentum == 'overbought':
                return 'high'
            elif momentum == 'oversold':
                return 'low'
                
            return 'medium'
            
        except Exception as e:
            self.logger.error(f"리스크 레벨 평가 중 오류 발생: {e}")
            return 'medium'

    def determine_market_state(self, analysis: Dict[str, Any]) -> str:
        """시장 상태 판단"""
        try:
            if not analysis:
                return 'unknown'
                
            # 변동성 기반 상태
            if analysis['volatility'] > self.volatility_thresholds['very_high']:
                return 'high_volatility'
            elif analysis['volatility'] < self.volatility_thresholds['very_low']:
                return 'low_volatility'
            
            # RSI 기반 상태
            if analysis['rsi']['state'] == 'overbought':
                return 'overbought'
            elif analysis['rsi']['state'] == 'oversold':
                return 'oversold'
            
            # 볼린저 밴드 기반 상태
            if analysis['bollinger']['state'] == 'overbought' and analysis['bollinger']['squeeze']:
                return 'overbought_squeeze'
            elif analysis['bollinger']['state'] == 'oversold' and analysis['bollinger']['squeeze']:
                return 'oversold_squeeze'
            
            # MACD 기반 상태
            if analysis['macd']['state'] == 'bullish' and analysis['macd']['hist_direction'] == 'up':
                return 'bullish_momentum'
            elif analysis['macd']['state'] == 'bearish' and analysis['macd']['hist_direction'] == 'down':
                return 'bearish_momentum'
            
            # 일목균형표 기반 상태
            if analysis['ichimoku']['cloud_state'] == 'above_cloud' and analysis['ichimoku']['conversion_base'] == 'bullish':
                return 'strong_uptrend'
            elif analysis['ichimoku']['cloud_state'] == 'below_cloud' and analysis['ichimoku']['conversion_base'] == 'bearish':
                return 'strong_downtrend'
            
            # 스토캐스틱 기반 상태
            if analysis['stochastic']['state'] == 'overbought' and analysis['stochastic']['cross'] == 'dead':
                return 'overbought_reversal'
            elif analysis['stochastic']['state'] == 'oversold' and analysis['stochastic']['cross'] == 'golden':
                return 'oversold_reversal'
            
            # ADX 기반 상태
            if analysis['adx']['trend_strength'] == 'strong' and analysis['adx']['trend_direction'] == 'up':
                return 'strong_uptrend'
            elif analysis['adx']['trend_strength'] == 'strong' and analysis['adx']['trend_direction'] == 'down':
                return 'strong_downtrend'
            
            return 'normal'
            
        except Exception as e:
            logger.error(f"시장 상태 판단 중 오류 발생: {e}")
            return 'unknown'

    def calculate_risk_level(self, analysis: Dict[str, Any]) -> str:
        """리스크 레벨 계산"""
        try:
            if not analysis:
                return 'unknown'
                
            if analysis['volatility'] > self.volatility_thresholds['high']:
                return 'very_high'
            elif analysis['volatility'] > self.volatility_thresholds['medium']:
                return 'high'
            elif analysis['rsi']['state'] in ['overbought', 'oversold']:
                return 'high'
            elif analysis['bollinger']['squeeze']:
                return 'high'
            elif analysis['macd']['state'] == 'bearish' and analysis['macd']['hist_direction'] == 'down':
                return 'high'
            elif analysis['adx']['trend_strength'] == 'weak':
                return 'medium'
            elif analysis['trend']['strength'] < 0.3:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"리스크 레벨 계산 중 오류 발생: {e}")
            return 'unknown'

    def get_market_state(self) -> str:
        """현재 시장 상태 반환"""
        try:
            if not hasattr(self, 'last_market_state'):
                return 'unknown'
            return self.last_market_state
        except Exception as e:
            logger.error(f"시장 상태 조회 중 오류 발생: {e}")
            return 'unknown' 