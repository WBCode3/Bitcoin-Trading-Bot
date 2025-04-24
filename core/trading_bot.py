import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import time
import asyncio
from .market_analyzer import MarketAnalyzer
from .strategy import TradingStrategy
from .risk_manager import RiskManager
from .exchange import Exchange
from config.settings import settings
from utils.logger import setup_logger
from .notifier import Notifier

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.exchange = Exchange(
            api_key=settings.API_KEY,
            api_secret=settings.API_SECRET
        )
        self.market_analyzer = MarketAnalyzer()
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManager()
        self.notifier = Notifier()
        
        self.is_running = False
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=5)
        self.health_check_interval = timedelta(minutes=30)
        self.last_health_check = None
        
        self.current_position = None
        self.trade_history = []
        
        self.last_status_update = None
        
        logger.info("트레이딩 봇 초기화 완료")
        
    async def start(self) -> None:
        """트레이딩 봇 시작"""
        try:
            if self.is_running:
                logger.warning("이미 실행 중인 트레이딩 봇입니다.")
                return
                
            self.is_running = True
            logger.info("트레이딩 봇 시작")
            
            # 시작 알림 전송
            self.notifier.send_telegram_message(
                f"<b>🚀 트레이딩 봇 시작</b>\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # 초기 계좌 상태 설정
            balance = self.exchange.get_balance()
            self.risk_manager.update_account_state(
                float(balance['total']),
                0.0
            )
            
            # 메인 루프 시작
            while self.is_running:
                try:
                    await self._trading_cycle()
                    await asyncio.sleep(1)  # CPU 부하 방지
                    
                except Exception as e:
                    logger.error(f"트레이딩 사이클 중 오류 발생: {e}")
                    self.notifier.send_error_alert(str(e))
                    await asyncio.sleep(5)  # 오류 발생 시 잠시 대기
                    
        except Exception as e:
            logger.error(f"트레이딩 봇 시작 중 오류 발생: {e}")
            self.notifier.send_error_alert(str(e))
            self.is_running = False
            
    async def stop(self) -> None:
        """트레이딩 봇 중지"""
        try:
            if not self.is_running:
                logger.warning("실행 중이지 않은 트레이딩 봇입니다.")
                return
                
            self.is_running = False
            logger.info("트레이딩 봇 중지")
            
            # 종료 알림 전송
            stats = self.get_trading_stats()
            self.notifier.send_telegram_message(
                f"<b>🛑 트레이딩 봇 종료</b>\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"총 거래 횟수: {stats['total_trades']}\n"
                f"승률: {stats['win_rate']:.2%}\n"
                f"총 수익: {stats['total_pnl']:.2%}"
            )
            
            # 현재 포지션 정리
            if self.current_position:
                await self._close_position(self.current_position)
                
        except Exception as e:
            logger.error(f"트레이딩 봇 중지 중 오류 발생: {e}")
            self.notifier.send_error_alert(str(e))
            
    async def _trading_cycle(self) -> None:
        """트레이딩 사이클 실행"""
        try:
            # 헬스 체크
            if (self.last_health_check is None or 
                datetime.now() - self.last_health_check >= self.health_check_interval):
                await self._health_check()
                
            # 시장 데이터 수집
            market_data = await self._collect_market_data()
            if not market_data:
                logger.warning("시장 데이터 수집 실패")
                return
                
            # 현재 가격과 지표 정보 콘솔 출력 (1분마다)
            if (self.last_status_update is None or 
                datetime.now() - self.last_status_update >= timedelta(minutes=1)):
                self._print_status(market_data)
                
                self.last_status_update = datetime.now()
                
            # 시장 분석
            analysis = self.market_analyzer.analyze_market_condition(market_data)
            if not analysis:
                logger.warning("시장 분석 실패")
                return
                
            # 리스크 한도 체크
            if not self.risk_manager.check_risk_limits():
                logger.warning("리스크 한도 초과로 트레이딩 중단")
                return
                
            # 현재 포지션 확인
            position = self.exchange.get_position()
            if position:
                # 청산 조건 체크
                should_close, close_pct = self.strategy.check_exit_conditions(
                    market_data,
                    position['side'],
                    position['entry_price']
                )
                
                if should_close:
                    await self._close_position(position, close_pct)
                    
            else:
                # 진입 신호 생성
                signal = self.strategy.generate_signal(market_data)
                if signal:
                    await self._execute_trade(signal, market_data)
                    
        except Exception as e:
            logger.error(f"트레이딩 사이클 실행 중 오류 발생: {e}")
            self.notifier.send_error_alert(f"트레이딩 사이클 오류: {str(e)}")
            
    async def _collect_market_data(self) -> Dict[str, Any]:
        """시장 데이터 수집"""
        try:
            # OHLCV 데이터 가져오기
            ohlcv = await self.exchange.get_ohlcv(
                symbol='BTCUSDT',
                interval='5m',
                limit=100
            )
            if not ohlcv:
                logger.warning("시장 데이터 수집 실패")
                return None
                
            # 현재 가격 가져오기
            current_price = float(ohlcv[-1][4])  # 마지막 캔들의 종가
            
            # 데이터프레임 생성
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 지표 계산
            indicators = {}
            
            # RSI 계산 (단기/장기)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 단기 RSI (7일)
            gain_short = (delta.where(delta > 0, 0)).rolling(window=7).mean()
            loss_short = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
            rs_short = gain_short / loss_short
            rsi_short = 100 - (100 / (1 + rs_short))
            
            # 장기 RSI (21일)
            gain_long = (delta.where(delta > 0, 0)).rolling(window=21).mean()
            loss_long = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
            rs_long = gain_long / loss_long
            rsi_long = 100 - (100 / (1 + rs_long))
            
            indicators['rsi'] = {
                'state': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral',
                'value': rsi.iloc[-1],
                'short_term': rsi_short.iloc[-1],
                'long_term': rsi_long.iloc[-1]
            }
            
            # ADX 분석
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
            minus_dm = (-low_diff).where((low_diff > 0) & (low_diff > high_diff), 0)
            
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            trend_strength = 'weak'
            if adx.iloc[-1] > 25:
                trend_strength = 'strong'
            elif adx.iloc[-1] > 20:
                trend_strength = 'moderate'
                
            trend_direction = 'up' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'down'
            
            indicators['adx'] = {
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'adx': float(adx.iloc[-1]),
                'plus_di': float(plus_di.iloc[-1]),
                'minus_di': float(minus_di.iloc[-1])
            }
            
            # 스토캐스틱 분석
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=3).mean()
            
            stoch_state = 'neutral'
            if k.iloc[-1] > 80 and d.iloc[-1] > 80:
                stoch_state = 'overbought'
            elif k.iloc[-1] < 20 and d.iloc[-1] < 20:
                stoch_state = 'oversold'
                
            cross = 'none'
            if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]:
                cross = 'bullish'
            elif k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]:
                cross = 'bearish'
                
            indicators['stochastic'] = {
                'state': stoch_state,
                'cross': cross,
                'k': float(k.iloc[-1]),
                'd': float(d.iloc[-1])
            }
            
            # 일목균형표 분석
            conversion = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
            base = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
            leading_span_a = ((conversion + base) / 2).shift(26)
            leading_span_b = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
            
            cloud_state = 'neutral'
            if current_price > leading_span_a.iloc[-1] and current_price > leading_span_b.iloc[-1]:
                cloud_state = 'above_cloud'
            elif current_price < leading_span_a.iloc[-1] and current_price < leading_span_b.iloc[-1]:
                cloud_state = 'below_cloud'
                
            conversion_base = 'bullish' if conversion.iloc[-1] > base.iloc[-1] else 'bearish'
            
            indicators['ichimoku'] = {
                'cloud_state': cloud_state,
                'conversion_base': conversion_base,
                'conversion': float(conversion.iloc[-1]),
                'base': float(base.iloc[-1]),
                'leading_span_a': float(leading_span_a.iloc[-1]),
                'leading_span_b': float(leading_span_b.iloc[-1])
            }
            
            # 거래량 분석
            volume_ma = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1]
            indicators['volume'] = {
                'ratio': float(volume_ratio),
                'trend': 'increasing' if volume_ratio > 1.2 else 'decreasing' if volume_ratio < 0.8 else 'neutral',
                'current': float(df['volume'].iloc[-1]),
                'ma': float(volume_ma.iloc[-1])
            }
            
            # 트렌드 분석
            ma_short = df['close'].rolling(window=20).mean()
            ma_medium = df['close'].rolling(window=50).mean()
            trend_strength = abs((ma_short.iloc[-1] - ma_medium.iloc[-1]) / ma_medium.iloc[-1])
            indicators['trend'] = {
                'direction': 'up' if ma_short.iloc[-1] > ma_medium.iloc[-1] else 'down',
                'strength': float(min(max(trend_strength, 0), 1))
            }
            
            # 볼린저 밴드 계산
            bb_width = self._calculate_bollinger_bands(df)['width']
            indicators['bollinger'] = {
                'state': 'squeeze' if bb_width < 0.02 else 'expansion',
                'width': bb_width,
                'upper': self._calculate_bollinger_bands(df)['upper'].iloc[-1],
                'lower': self._calculate_bollinger_bands(df)['lower'].iloc[-1]
            }
            
            # MACD 계산
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            indicators['macd'] = {
                'state': 'bullish' if histogram.iloc[-1] > 0 else 'bearish',
                'histogram': histogram.iloc[-1],
                'macd_line': macd.iloc[-1],
                'signal_line': signal.iloc[-1]
            }
            
            # 변동성 계산
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            indicators['volatility'] = float(volatility) if not np.isnan(volatility) else 0.02
            
            # 시장 데이터 생성
            market_data = {
                'current_price': current_price,
                'indicators': indicators,
                'risk_level': 'medium'
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"시장 데이터 수집 중 오류 발생: {e}")
            return None
            
    async def _execute_trade(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> None:
        """거래 실행"""
        try:
            # 거래 간격 체크
            current_time = datetime.now()
            if (self.last_trade_time is not None and 
                current_time - self.last_trade_time < self.min_trade_interval):
                return
                
            # 포지션 크기 계산
            position_size, leverage = self.risk_manager.calculate_position_size(market_data)
            
            if position_size <= 0:
                logger.warning("포지션 크기가 0 이하입니다.")
                return
                
            # 레버리지 설정
            self.exchange.set_leverage(leverage)
            
            # 주문 실행
            order = self.exchange.create_order(
                symbol=self.exchange.symbol,
                side=signal['type'],
                quantity=position_size,
                order_type='MARKET'
            )
            
            if not order:
                logger.error("주문 실행 실패")
                return
                
            # 포지션 정보 업데이트
            self.current_position = {
                'side': signal['type'],
                'entry_price': float(order['price']),
                'size': position_size,
                'leverage': leverage,
                'timestamp': current_time
            }
            
            self.last_trade_time = current_time
            logger.info(f"새로운 포지션 진입: {self.current_position}")
            
        except Exception as e:
            logger.error(f"거래 실행 중 오류 발생: {e}")
            
    async def _close_position(self, position: Dict[str, Any], close_pct: float = 1.0) -> None:
        """포지션 청산"""
        try:
            if close_pct <= 0:
                return
                
            # 청산 주문 실행
            close_size = position['size'] * close_pct
            order = await self.exchange.create_order(
                'sell' if position['side'] == 'buy' else 'buy',
                close_size
            )
            
            if not order:
                logger.error("청산 주문 실행 실패")
                return
                
            # 손익 계산
            entry_price = position['entry_price']
            exit_price = float(order['price'])
            
            if position['side'] == 'buy':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
                
            pnl *= position['leverage']
            
            # 거래 기록 업데이트
            trade_info = {
                'side': position['side'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': close_size,
                'leverage': position['leverage'],
                'pnl': pnl,
                'timestamp': datetime.now()
            }
            
            self.trade_history.append(trade_info)
            
            # 리스크 관리자 업데이트
            self.risk_manager.update_trade_result(pnl > 0, pnl)
            
            # 포지션 정보 업데이트
            if close_pct == 1.0:
                self.current_position = None
            else:
                self.current_position['size'] -= close_size
                
            logger.info(f"포지션 {close_pct*100}% 청산: {trade_info}")
            
        except Exception as e:
            logger.error(f"포지션 청산 중 오류 발생: {e}")
            
    async def _health_check(self) -> None:
        """시스템 상태 점검"""
        try:
            # 계좌 잔고 확인
            balance = self.exchange.get_balance()
            if not balance:
                logger.error("계좌 잔고 확인 실패")
                return
                
            # 포지션 상태 확인
            positions = self.exchange.get_positions()
            if positions:
                for position in positions:
                    if float(position['size']) > 0:
                        logger.warning(f"미청산 포지션 발견: {position}")
                        
            # API 연결 상태 확인
            try:
                await self.exchange.get_ohlcv('BTCUSDT', '1m', limit=1)
            except Exception as e:
                logger.error(f"API 연결 상태 확인 실패: {e}")
                return
                
            # 리스크 한도 체크
            if not self.risk_manager.check_risk_limits():
                logger.warning("리스크 한도 초과")
                
            self.last_health_check = datetime.now()
            logger.info("시스템 상태 점검 완료")
            
        except Exception as e:
            logger.error(f"시스템 상태 점검 중 오류 발생: {e}")
            
    def get_trading_stats(self) -> Dict[str, Any]:
        """트레이딩 통계 조회"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'total_pnl': 0.0
                }
                
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            total_pnl = sum(trade['pnl'] for trade in self.trade_history)
            
            return {
                'total_trades': total_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
                'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0.0,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            logger.error(f"트레이딩 통계 조회 중 오류 발생: {e}")
            return None

    def execute_trade(self) -> None:
        """거래 실행"""
        try:
            # 시장 데이터 생성
            market_data = {
                'indicators': {
                    'rsi': {'state': None, 'short_term': None, 'long_term': None},
                    'bollinger': {'state': None, 'squeeze': None},
                    'macd': {'state': None, 'hist_direction': None},
                    'ichimoku': {'cloud_state': None, 'conversion_base': None},
                    'stochastic': {'state': None, 'cross': None},
                    'adx': {'trend_strength': None, 'trend_direction': None},
                    'volatility': None
                },
                'risk_level': 'medium'
            }
            
            # 전략 실행
            self.strategy.execute_trade(market_data)
            
        except Exception as e:
            logger.error(f"거래 실행 중 오류 발생: {e}")
            raise 

    def _print_status(self, market_data: Dict[str, Any]) -> None:
        """상태 출력"""
        try:
            indicators = market_data['indicators']
            current_price = market_data['current_price']
            
            # RSI 수치
            rsi_short = indicators['rsi']['short_term']
            rsi_long = indicators['rsi']['long_term']
            
            # 볼린저 밴드 수치
            bb_width = indicators['bollinger']['width']
            bb_upper = indicators['bollinger']['upper']
            bb_lower = indicators['bollinger']['lower']
            
            # MACD 수치
            macd_line = indicators['macd']['macd_line']
            macd_signal = indicators['macd']['signal_line']
            macd_hist = indicators['macd']['histogram']
            
            # 스토캐스틱 수치
            stoch_k = indicators['stochastic']['k']
            stoch_d = indicators['stochastic']['d']
            
            # ADX 수치
            adx = indicators['adx']['adx']
            plus_di = indicators['adx']['plus_di']
            minus_di = indicators['adx']['minus_di']
            
            # 일목균형표 수치
            conversion = indicators['ichimoku']['conversion']
            base = indicators['ichimoku']['base']
            leading_span_a = indicators['ichimoku']['leading_span_a']
            leading_span_b = indicators['ichimoku']['leading_span_b']
            
            # 거래량 수치
            volume = indicators['volume']['current']
            volume_ma = indicators['volume']['ma']
            volume_ratio = indicators['volume']['ratio']
            
            print("\n==================================================")
            print(f"🕒 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💰 현재 가격: ${current_price:,.2f}")
            print("\n📊 RSI:")
            print(f"  - 단기: {rsi_short:.2f}")
            print(f"  - 장기: {rsi_long:.2f}")
            
            print("\n📈 볼린저 밴드:")
            print(f"  - 상단: ${bb_upper:,.2f}")
            print(f"  - 하단: ${bb_lower:,.2f}")
            print(f"  - 폭: {bb_width:.4f}")
            
            print("\n📉 MACD:")
            print(f"  - MACD: {macd_line:.2f}")
            print(f"  - 시그널: {macd_signal:.2f}")
            print(f"  - 히스토그램: {macd_hist:.2f}")
            
            print("\n📊 스토캐스틱:")
            print(f"  - %K: {stoch_k:.2f}")
            print(f"  - %D: {stoch_d:.2f}")
            
            print("\n📈 ADX:")
            print(f"  - ADX: {adx:.2f}")
            print(f"  - +DI: {plus_di:.2f}")
            print(f"  - -DI: {minus_di:.2f}")
            
            print("\n📊 일목균형표:")
            print(f"  - 전환선: ${conversion:,.2f}")
            print(f"  - 기준선: ${base:,.2f}")
            print(f"  - 선행스팬A: ${leading_span_a:,.2f}")
            print(f"  - 선행스팬B: ${leading_span_b:,.2f}")
            
            print("\n📊 거래량:")
            print(f"  - 현재: {volume:,.0f}")
            print(f"  - 이동평균: {volume_ma:,.0f}")
            print(f"  - 비율: {volume_ratio:.2f}")
            
            print("\n⚡ 변동성: {:.2f}%".format(indicators['volatility'] * 100))
            print(f"⚠️ 리스크 레벨: {market_data['risk_level']}")
            print("==================================================\n")
            
        except Exception as e:
            logger.error(f"상태 출력 중 오류 발생: {e}") 

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            if len(data) < period:
                logger.warning(f"볼린저 밴드 계산을 위한 데이터가 부족합니다. (필요: {period}, 현재: {len(data)})")
                current_price = data['close'].iloc[-1]
                return {
                    'middle': pd.Series([current_price] * len(data)),
                    'upper': pd.Series([current_price * 1.02] * len(data)),
                    'lower': pd.Series([current_price * 0.98] * len(data))
                }
            
            sma = data['close'].rolling(window=period, min_periods=1).mean()
            std_dev = data['close'].rolling(window=period, min_periods=1).std()
            
            # NaN 값 처리
            current_price = data['close'].iloc[-1]
            sma = sma.fillna(current_price)
            std_dev = std_dev.fillna(0)
            
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            
            return {'middle': sma, 'upper': upper, 'lower': lower}
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 중 오류 발생: {e}")
            current_price = data['close'].iloc[-1]
            return {
                'middle': pd.Series([current_price] * len(data)),
                'upper': pd.Series([current_price * 1.02] * len(data)),
                'lower': pd.Series([current_price * 0.98] * len(data))
            } 