import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from decimal import Decimal
import os
import ta

from core.exchange import Exchange
from core.risk_manager import RiskManager
from core.strategy import TradingStrategy
from core.market_analyzer import MarketAnalyzer

class BacktestSimulator:
    def __init__(self, initial_capital: float = 10000.0, risk_manager=None, trading_strategy=None, market_analyzer=None):
        self.initial_capital = Decimal(str(initial_capital))
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.daily_results = []
        self.max_drawdown = Decimal('0')
        self.capital_history = []
        
        # 컴포넌트 초기화
        self.exchange = Exchange()
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        self.strategy = trading_strategy if trading_strategy else TradingStrategy()
        self.market_analyzer = market_analyzer if market_analyzer else MarketAnalyzer()
        
        # 로거 설정
        self.logger = logging.getLogger('backtest_simulator')
        self.logger.setLevel(logging.INFO)
        
        # 이미 핸들러가 있는지 확인하고 없으면 추가
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            # 타임스탬프를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 필요한 컬럼만 선택
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_columns]
            
            # 데이터 타입 변환 및 결측값 처리
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # 결측값이 있는 경우 경고 로그
            if df.isna().any().any():
                self.logger.warning("데이터에 결측값이 있습니다. 전방/후방 채우기로 처리했습니다.")
            
            # 기술적 지표 계산
            df['rsi'] = self._calculate_rsi(df)
            stoch_k, stoch_d = self._calculate_stochastic(df)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # 볼린저 밴드 계산
            bb = self._calculate_bollinger_bands(df)
            if bb is not None:
                df['bb_upper'] = bb['upper'].fillna(method='ffill').fillna(method='bfill')
                df['bb_middle'] = bb['middle'].fillna(method='ffill').fillna(method='bfill')
                df['bb_lower'] = bb['lower'].fillna(method='ffill').fillna(method='bfill')
            else:
                # 볼린저 밴드 계산 실패 시 기본값 설정
                current_price = df['close'].iloc[-1]
                df['bb_upper'] = current_price * 1.02
                df['bb_middle'] = current_price
                df['bb_lower'] = current_price * 0.98
            
            macd = self._calculate_macd(df)
            df['macd'] = macd['line']
            df['macd_signal'] = macd['signal']
            df['macd_hist'] = macd['histogram']
            
            df['adx'] = self._calculate_adx(df)
            
            # 변동성 계산 (ATR)
            df['volatility'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # 거래량 지표
            volume_indicators = self._calculate_volume_indicators(df)
            df['volume_ratio'] = volume_indicators['ratio']
            
            # 거래량 이동평균
            df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {e}")
            raise

    def _initialize_simulation(self):
        """시뮬레이션 초기화"""
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.daily_results = []
        self.max_drawdown = Decimal('0')
        self.capital_history = [self.initial_capital]
        
    def run_simulation(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> None:
        """시뮬레이션 실행"""
        try:
            # 시뮬레이션 초기화
            self._initialize_simulation()
            
            # 데이터 로드 및 전처리
            df = self._prepare_market_data()
            
            # 날짜 필터링
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            # 데이터가 충분한지 확인
            if len(df) < 20:
                raise ValueError("시뮬레이션을 위한 충분한 데이터가 없습니다.")
            
            # 시뮬레이션 진행
            for i in range(20, len(df)):
                current_data = self._collect_market_data(df, i)
                signal = self._generate_signal(current_data)
                
                if signal['action'] != 'hold':
                    self._manage_positions(current_data, signal)
                
                # 일일 결과 기록
                if i % 96 == 0:  # 15분 간격으로 하루는 96개의 데이터 포인트
                    self._record_daily_result(current_data['timestamp'])
                
            # 최종 결과 계산
            self._calculate_final_results()
            
        except Exception as e:
            self.logger.error(f"시뮬레이션 실행 중 오류 발생: {e}")
            raise

    def _collect_market_data(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """시장 데이터 수집"""
        try:
            if current_index < 100:  # 최소 데이터 포인트 필요
                return {
                    'current_price': float(df['close'].iloc[current_index]),
                    'timestamp': df['timestamp'].iloc[current_index],
                    'indicators': {},
                    'analysis': {},
                    'risk_level': 'medium'
                }
                
            # 과거 데이터 슬라이스
            historical_data = df.iloc[max(0, current_index-100):current_index+1]
            
            # 현재 가격
            current_price = float(historical_data['close'].iloc[-1])
            current_timestamp = historical_data['timestamp'].iloc[-1]
            
            # 지표 계산
            indicators = {
                'rsi': float(historical_data['rsi'].iloc[-1]),
                'macd': float(historical_data['macd'].iloc[-1]),
                'macd_signal': float(historical_data['macd_signal'].iloc[-1]),
                'bb_upper': float(historical_data['bb_upper'].iloc[-1]),
                'bb_middle': float(historical_data['bb_middle'].iloc[-1]),
                'bb_lower': float(historical_data['bb_lower'].iloc[-1]),
                'stoch_k': float(historical_data['stoch_k'].iloc[-1]),
                'stoch_d': float(historical_data['stoch_d'].iloc[-1]),
                'adx': float(historical_data['adx'].iloc[-1]),
                'volatility': float(historical_data['volatility'].iloc[-1]),
                'volume': float(historical_data['volume'].iloc[-1]),
                'volume_ratio': float(historical_data['volume_ratio'].iloc[-1])
            }
            
            # 시장 분석
            market_analysis = {
                'trend': self._analyze_trend(historical_data),
                'strength': self._analyze_strength(historical_data),
                'volume': self._analyze_volume(historical_data)
            }
            
            return {
                'current_price': current_price,
                'timestamp': current_timestamp,
                'indicators': indicators,
                'analysis': market_analysis,
                'risk_level': self._analyze_risk(indicators)
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 중 오류 발생: {e}")
            return {
                'current_price': float(df['close'].iloc[current_index]),
                'timestamp': df['timestamp'].iloc[current_index],
                'indicators': {},
                'analysis': {},
                'risk_level': 'medium'
            }

    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """추세 분석"""
        try:
            last_price = data['close'].iloc[-1]
            sma20 = data['close'].rolling(window=20).mean().iloc[-1]
            sma50 = data['close'].rolling(window=50).mean().iloc[-1]
            
            if last_price > sma20 and sma20 > sma50:
                return 'uptrend'
            elif last_price < sma20 and sma20 < sma50:
                return 'downtrend'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def _analyze_strength(self, data: pd.DataFrame) -> str:
        """추세 강도 분석"""
        try:
            adx = data['adx'].iloc[-1]
            if adx > 30:
                return 'strong'
            elif adx > 20:
                return 'moderate'
            else:
                return 'weak'
        except:
            return 'moderate'

    def _analyze_volume(self, data: pd.DataFrame) -> str:
        """거래량 분석"""
        try:
            volume_ratio = data['volume_ratio'].iloc[-1]
            if volume_ratio > 1.5:
                return 'high'
            elif volume_ratio < 0.5:
                return 'low'
            else:
                return 'normal'
        except:
            return 'normal'

    def _analyze_risk(self, indicators: Dict[str, float]) -> str:
        """리스크 레벨 분석"""
        try:
            volatility = indicators['volatility']
            if volatility > 25:
                return 'high'
            elif volatility > 15:
                return 'medium'
            else:
                return 'low'
        except:
            return 'medium'

    def _manage_positions(self, current_data: Dict[str, Any], signal: Dict[str, Any]) -> None:
        """포지션 관리"""
        try:
            # 현재 포지션 확인 및 관리
            for position in self.positions[:]:  # 리스트 복사본으로 반복
                # 현재 가격
                current_price = float(current_data['current_price'])
                
                # 포지션 진입가
                entry_price = float(position['entry_price'])
                
                # PNL 계산
                pnl = (current_price - entry_price) * position['size']
                if position['side'] == 'short':
                    pnl = -pnl
                
                # 변동성 기반 손절/익절 계산
                volatility = current_data['indicators'].get('volatility', 0.02)
                stop_loss = volatility * 2.5  # 변동성의 2.5배
                take_profit = volatility * 3.0  # 변동성의 3.0배
                
                # 최대 드로다운 체크 (더 엄격한 조건)
                current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
                if current_drawdown > Decimal('0.15'):  # 15% 이상 손실 시 모든 포지션 청산
                    self._close_position(position, current_data, 'max_drawdown')
                    continue
                
                # 연속 손실 체크 (더 엄격한 조건)
                if len(self.trades) >= 2:
                    last_two_trades = self.trades[-2:]
                    if all(t.get('pnl', 0) < 0 for t in last_two_trades):
                        self._close_position(position, current_data, 'consecutive_loss')
                        continue
                
                # 손절/익절 체크 (변동성 기반)
                if pnl < -stop_loss:
                    self._close_position(position, current_data, 'stop_loss')
                    continue
                
                if pnl > take_profit:
                    self._close_position(position, current_data, 'take_profit')
                    continue
            
            # 새로운 포지션 진입 검토
            if signal['action'] != 'hold' and not self.positions:  # 시그널이 있고 현재 포지션이 없을 때
                # 변동성 기반 포지션 크기 조정
                volatility = current_data['indicators'].get('volatility', 0.02)
                base_size = 0.1
                
                # 변동성에 따른 포지션 크기 조정 (더 보수적인 접근)
                if volatility > 0.03:  # 변동성이 높을 때
                    position_size = base_size * 0.3
                elif volatility < 0.01:  # 변동성이 낮을 때
                    position_size = base_size * 1.2
                else:
                    position_size = base_size
                
                # 신뢰도에 따른 포지션 크기 조정
                confidence = signal.get('confidence', 0.0)
                position_size *= confidence
                
                # 레버리지 동적 조정 (더 보수적인 접근)
                leverage = min(int(8 / (volatility * 100)), 8)  # 최대 8배
                
                if signal['action'] == 'long':
                    self._open_position('long', position_size, leverage, float(current_data['current_price']))
                elif signal['action'] == 'short':
                    self._open_position('short', position_size, leverage, float(current_data['current_price']))
            
        except Exception as e:
            self.logger.error(f"포지션 관리 중 오류 발생: {str(e)}")

    def _execute_trade(self, market_data: Dict[str, Any], signal: Dict[str, Any]) -> None:
        """매매 실행"""
        try:
            if not signal:
                return
                
            # 포지션 크기 계산
            position_size, leverage = self.risk_manager.calculate_position_size(
                market_data.get('indicators', {}),
                market_data.get('current_price', 0.0)
            )
            
            # 리스크 한도 체크
            if not self.risk_manager.check_risk_limits(market_data):
                self.logger.warning("리스크 한도 초과로 매매 중단")
                return
                
            # 매매 실행
            if signal['action'] == 'buy':
                self._open_position('long', position_size, leverage, market_data['current_price'])
            elif signal['action'] == 'sell':
                self._open_position('short', position_size, leverage, market_data['current_price'])
                
        except Exception as e:
            self.logger.error(f"매매 실행 중 오류 발생: {e}")

    def _open_position(self, side: str, size: float, leverage: int, price: float) -> None:
        """포지션 오픈"""
        try:
            # 포지션 정보 저장
            position = {
                'side': side,
                'size': size,
                'leverage': leverage,
                'entry_price': price,
                'entry_time': datetime.now()
            }
            
            # 포지션 리스트에 추가
            self.positions.append(position)
            
            # 거래 기록
            self.trades.append({
                'type': 'open',
                'side': side,
                'size': size,
                'leverage': leverage,
                'price': price,
                'time': datetime.now()
            })
            
            self.logger.info(f"새로운 포지션 오픈: {side} {size} @ {price}")
            
        except Exception as e:
            self.logger.error(f"포지션 오픈 중 오류 발생: {e}")

    def _close_position(self, position: Dict[str, Any], current_data: Dict[str, Any], reason: str) -> None:
        """포지션 청산"""
        try:
            # 수익/손실 계산
            entry_price = Decimal(str(position['entry_price']))
            current_price = Decimal(str(current_data['current_price']))
            position_size = Decimal(str(position['size']))
            
            if position['side'] == 'long':
                pnl = (current_price - entry_price) * position_size
            else:  # short
                pnl = (entry_price - current_price) * position_size
            
            # 자본금 업데이트
            self.current_capital += pnl
            
            # 거래 기록 추가
            trade_record = {
                'entry_price': float(entry_price),
                'exit_price': float(current_price),
                'size': float(position_size),
                'side': position['side'],
                'pnl': float(pnl),
                'reason': reason,
                'type': 'close'
            }
            self.trades.append(trade_record)
            
            # 포지션 목록에서 제거
            self.positions.remove(position)
            
            self.logger.info(f"포지션 청산: {position['side']} {float(position_size)} @ {float(current_price)} (PNL: {float(pnl):.2f})")
            
        except Exception as e:
            self.logger.error(f"포지션 청산 중 오류 발생: {str(e)}")
            raise
            
    def _record_daily_result(self, current_time: datetime):
        """일일 결과 기록"""
        try:
            # 자본금 기록
            self.capital_history.append(self.current_capital)
            
            # 최대 낙폭 계산
            peak = max(self.capital_history)
            drawdown = (peak - self.current_capital) / peak
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # 일일 결과 저장
            self.daily_results.append({
                'date': current_time,
                'capital': self.current_capital,
                'drawdown': drawdown,
                'positions': len(self.positions)
            })
            
        except Exception as e:
            self.logger.error(f"일일 결과 기록 중 오류 발생: {e}")
            
    def _calculate_final_results(self):
        """최종 결과 계산"""
        try:
            # 총 수익률
            if self.initial_capital > 0:
                total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            else:
                total_return = Decimal('0')
                
            # 연간 수익률
            if len(self.daily_results) > 1:
                days = (self.daily_results[-1]['date'] - self.daily_results[0]['date']).days
                if days > 0:
                    annual_return = (Decimal('1') + total_return) ** (Decimal('365') / Decimal(str(days))) - Decimal('1')
                else:
                    annual_return = Decimal('0')
            else:
                annual_return = Decimal('0')
                
            # 승률 계산 수정 (type이 'close'인 거래만 계산)
            closed_trades = [t for t in self.trades if t.get('type') == 'close']
            winning_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
            total_trades = len(closed_trades)
            win_rate = Decimal(str(winning_trades)) / Decimal(str(total_trades)) if total_trades > 0 else Decimal('0')
            
            # 결과 출력
            self.logger.info(f"시뮬레이션 결과:")
            self.logger.info(f"초기 자본금: {float(self.initial_capital):,.2f}")
            self.logger.info(f"최종 자본금: {float(self.current_capital):,.2f}")
            self.logger.info(f"총 수익률: {float(total_return):.2%}")
            self.logger.info(f"연간 수익률: {float(annual_return):.2%}")
            self.logger.info(f"최대 낙폭: {float(self.max_drawdown):.2%}")
            self.logger.info(f"총 거래 횟수: {total_trades}")
            self.logger.info(f"승률: {float(win_rate):.2%}")
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'max_drawdown': float(self.max_drawdown),
                'total_trades': total_trades,
                'win_rate': float(win_rate)
            }
            
        except Exception as e:
            self.logger.error(f"최종 결과 계산 중 오류 발생: {e}")
            return {}

    def prepare_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if data is None or len(data) == 0:
                self.logger.warning("데이터가 비어있습니다.")
                return None

            # 현재 가격 설정
            current_price = float(data['close'].iloc[-1])
            
            market_data = {
                'current_price': current_price,
                'price': current_price,
                'indicators': {}
            }

            # 기본 지표 계산
            try:
                rsi = self._calculate_rsi(data)
                market_data['indicators']['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
            except Exception as e:
                self.logger.warning(f"RSI 계산 중 오류: {str(e)}")
                market_data['indicators']['rsi'] = 50

            try:
                stoch_k, stoch_d = self._calculate_stochastic(data)
                market_data['indicators']['stoch_k'] = stoch_k.iloc[-1] if not stoch_k.empty else 50
                market_data['indicators']['stoch_d'] = stoch_d.iloc[-1] if not stoch_d.empty else 50
            except Exception as e:
                self.logger.warning(f"스토캐스틱 계산 중 오류: {str(e)}")
                market_data['indicators']['stoch_k'] = 50
                market_data['indicators']['stoch_d'] = 50

            try:
                bollinger = self._calculate_bollinger_bands(data)
                market_data['indicators']['bollinger'] = {
                    'upper': bollinger.get('upper', current_price * 1.02),
                    'middle': bollinger.get('middle', current_price),
                    'lower': bollinger.get('lower', current_price * 0.98)
                }
            except Exception as e:
                self.logger.warning(f"볼린저 밴드 계산 중 오류: {str(e)}")
                market_data['indicators']['bollinger'] = {
                    'upper': current_price * 1.02,
                    'middle': current_price,
                    'lower': current_price * 0.98
                }

            try:
                macd = self._calculate_macd(data)
                market_data['indicators']['macd'] = {
                    'macd': macd.get('line', 0),
                    'signal': macd.get('signal', 0),
                    'histogram': macd.get('histogram', 0)
                }
            except Exception as e:
                self.logger.warning(f"MACD 계산 중 오류: {str(e)}")
                market_data['indicators']['macd'] = {
                    'macd': 0,
                    'signal': 0,
                    'histogram': 0
                }

            try:
                adx = self._calculate_adx(data)
                market_data['indicators']['adx'] = adx.iloc[-1] if not adx.empty else 25
            except Exception as e:
                self.logger.warning(f"ADX 계산 중 오류: {str(e)}")
                market_data['indicators']['adx'] = 25

            try:
                ichimoku = self._calculate_ichimoku(data)
                market_data['indicators']['ichimoku'] = {
                    'tenkan': ichimoku.get('tenkan', current_price),
                    'kijun': ichimoku.get('kijun', current_price),
                    'senkou_a': ichimoku.get('senkou_a', current_price),
                    'senkou_b': ichimoku.get('senkou_b', current_price)
                }
            except Exception as e:
                self.logger.warning(f"일목균형표 계산 중 오류: {str(e)}")
                market_data['indicators']['ichimoku'] = {
                    'tenkan': current_price,
                    'kijun': current_price,
                    'senkou_a': current_price,
                    'senkou_b': current_price
                }

            try:
                volume = self._calculate_volume_indicators(data)
                market_data['indicators']['volume'] = {
                    'obv': volume.get('obv', 0),
                    'volume_ma': volume.get('volume_ma', 0)
                }
            except Exception as e:
                self.logger.warning(f"거래량 지표 계산 중 오류: {str(e)}")
                market_data['indicators']['volume'] = {
                    'obv': 0,
                    'volume_ma': 0
                }

            try:
                volatility = self._calculate_volatility(data)
                market_data['indicators']['volatility'] = volatility if volatility is not None else 0.01
            except Exception as e:
                self.logger.warning(f"변동성 계산 중 오류: {str(e)}")
                market_data['indicators']['volatility'] = 0.01

            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 준비 중 오류 발생: {str(e)}")
            return None

    def _calculate_rsi(self, data, period=14):
        try:
            close_prices = data['close'].astype(float)
            delta = close_prices.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # 초기값을 50으로 설정
        except Exception as e:
            self.logger.error(f"RSI 계산 중 오류 발생: {str(e)}")
            return pd.Series([50] * len(data))  # 오류 발생시 기본값 반환

    def _calculate_stochastic(self, data, k_period=14, d_period=3):
        try:
            if len(data) < k_period:
                return pd.Series([50] * len(data)), pd.Series([50] * len(data))
            
            # 데이터 전처리 및 유효성 검사
            close_prices = pd.to_numeric(data['close'], errors='coerce')
            low_prices = pd.to_numeric(data['low'], errors='coerce')
            high_prices = pd.to_numeric(data['high'], errors='coerce')
            
            # NaN 값 처리
            close_prices = close_prices.ffill().bfill()
            low_prices = low_prices.ffill().bfill()
            high_prices = high_prices.ffill().bfill()
            
            # 0으로 나누기 방지
            lowest_low = low_prices.rolling(window=k_period).min()
            highest_high = high_prices.rolling(window=k_period).max()
            
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.finfo(float).eps)  # 0을 작은 값으로 대체
            
            k_line = 100 * ((close_prices - lowest_low) / denominator)
            d_line = k_line.rolling(window=d_period).mean()
            
            # 범위 제한 (0-100)
            k_line = k_line.clip(0, 100)
            d_line = d_line.clip(0, 100)
            
            # NaN 값을 50으로 대체
            k_line = k_line.fillna(50)
            d_line = d_line.fillna(50)
            
            return k_line, d_line
        except Exception as e:
            self.logger.error(f"스토캐스틱 계산 중 오류 발생: {str(e)}")
            return pd.Series([50] * len(data)), pd.Series([50] * len(data))

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            # 데이터 유효성 검사
            if data is None or data.empty:
                self.logger.error("볼린저 밴드 계산을 위한 데이터가 없습니다.")
                return {
                    'middle': pd.Series([0.0]),
                    'upper': pd.Series([0.0]),
                    'lower': pd.Series([0.0])
                }

            if len(data) < period:
                self.logger.warning(f"볼린저 밴드 계산을 위한 데이터가 부족합니다. (필요: {period}, 현재: {len(data)})")
                current_price = data['close'].iloc[-1] if not data.empty else 0.0
                return {
                    'middle': pd.Series([current_price] * len(data)),
                    'upper': pd.Series([current_price * 1.02] * len(data)),
                    'lower': pd.Series([current_price * 0.98] * len(data))
                }
            
            # 종가 데이터를 float로 변환하고 NaN 값 처리
            close_prices = pd.to_numeric(data['close'], errors='coerce')
            if close_prices.isna().all():
                self.logger.error("유효한 종가 데이터가 없습니다.")
                return {
                    'middle': pd.Series([0.0] * len(data)),
                    'upper': pd.Series([0.0] * len(data)),
                    'lower': pd.Series([0.0] * len(data))
                }
            
            # NaN 값 처리 (전방 채우기 -> 후방 채우기 -> 0으로 채우기)
            close_prices = close_prices.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 이동평균과 표준편차 계산
            sma = close_prices.rolling(window=period, min_periods=1).mean()
            std_dev = close_prices.rolling(window=period, min_periods=1).std()
            
            # NaN 값 처리
            sma = sma.fillna(method='ffill').fillna(method='bfill').fillna(0)
            std_dev = std_dev.fillna(0)
            
            # 볼린저 밴드 계산
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            
            # 결과 유효성 검사
            if upper.isna().any() or lower.isna().any():
                self.logger.warning("볼린저 밴드 계산 결과에 NaN 값이 있습니다.")
                upper = upper.fillna(method='ffill').fillna(method='bfill').fillna(0)
                lower = lower.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return {
                'middle': sma,
                'upper': upper,
                'lower': lower
            }
            
        except Exception as e:
            self.logger.error(f"볼린저 밴드 계산 중 오류 발생: {e}")
            return {
                'middle': pd.Series([0.0] * len(data)),
                'upper': pd.Series([0.0] * len(data)),
                'lower': pd.Series([0.0] * len(data))
            }

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """MACD 계산"""
        try:
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            return {
                'line': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(histogram.iloc[-1]),
                'state': 'bullish' if histogram.iloc[-1] > 0 else 'bearish'
            }
        except Exception:
            return {
                'line': 0.0,
                'signal': 0.0,
                'histogram': 0.0,
                'state': 'neutral'
            }

    def _calculate_adx(self, data: pd.DataFrame, period=14) -> pd.Series:
        """ADX 계산"""
        try:
            if len(data) < period:
                return pd.Series([25] * len(data))  # 기본값 반환

            # True Range 계산
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            tr = pd.Series(np.maximum.reduce([tr1, tr2, tr3]))
            tr[0] = tr[1]  # 첫 번째 값 처리
            
            # Directional Movement
            up_move = pd.Series(high - np.roll(high, 1))
            down_move = pd.Series(np.roll(low, 1) - low)
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothing
            tr_14 = tr.rolling(window=period).mean()
            plus_di_14 = pd.Series(plus_dm).rolling(window=period).mean()
            minus_di_14 = pd.Series(minus_dm).rolling(window=period).mean()
            
            # DI 계산
            plus_di = 100 * (plus_di_14 / tr_14)
            minus_di = 100 * (minus_di_14 / tr_14)
            
            # ADX 계산
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = pd.Series(dx).rolling(window=period).mean()
            
            return adx.fillna(25)  # 결측값을 25로 채움
            
        except Exception as e:
            self.logger.error(f"ADX 계산 중 오류 발생: {e}")
            return pd.Series([25] * len(data))  # 오류 발생 시 기본값 반환

    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, Any]:
        """일목균형표 계산"""
        try:
            if len(data) < 52:  # 최소 데이터 포인트 체크
                return {
                    'conversion': 0.0,
                    'base': 0.0,
                    'leading_span1': 0.0,
                    'leading_span2': 0.0,
                    'cloud_state': 'neutral',
                    'conversion_base': 'neutral'
                }

            # 데이터를 numpy 배열로 변환
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # 전환선 (9일)
            high_9 = pd.Series(high).rolling(window=9, min_periods=1).max().values
            low_9 = pd.Series(low).rolling(window=9, min_periods=1).min().values
            conversion = (high_9 + low_9) / 2
            
            # 기준선 (26일)
            high_26 = pd.Series(high).rolling(window=26, min_periods=1).max().values
            low_26 = pd.Series(low).rolling(window=26, min_periods=1).min().values
            base = (high_26 + low_26) / 2
            
            # 선행스팬1 (전환선 + 기준선) / 2
            leading_span1 = (conversion + base) / 2
            
            # 선행스팬2 (52일 고가 + 저가) / 2
            high_52 = pd.Series(high).rolling(window=52, min_periods=1).max().values
            low_52 = pd.Series(low).rolling(window=52, min_periods=1).min().values
            leading_span2 = (high_52 + low_52) / 2
            
            # 현재 가격과 구름대 비교
            current_price = close[-1]
            cloud_state = 'neutral'
            if current_price > leading_span1[-1] and current_price > leading_span2[-1]:
                cloud_state = 'above_cloud'
            elif current_price < leading_span1[-1] and current_price < leading_span2[-1]:
                cloud_state = 'below_cloud'
                
            return {
                'conversion': float(conversion[-1]),
                'base': float(base[-1]),
                'leading_span1': float(leading_span1[-1]),
                'leading_span2': float(leading_span2[-1]),
                'cloud_state': cloud_state,
                'conversion_base': 'bullish' if conversion[-1] > base[-1] else 'bearish'
            }
        except Exception as e:
            self.logger.error(f"일목균형표 계산 중 오류 발생: {e}")
            return {
                'conversion': 0.0,
                'base': 0.0,
                'leading_span1': 0.0,
                'leading_span2': 0.0,
                'cloud_state': 'neutral',
                'conversion_base': 'neutral'
            }

    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래량 지표 계산"""
        try:
            if len(data) < 20:  # 최소 데이터 포인트 체크
                return {
                    'current': 0.0,
                    'ma': 0.0,
                    'ratio': 1.0,
                    'trend': 'neutral'
                }

            # 거래량 데이터를 numpy 배열로 변환
            volume = data['volume'].values
            
            # 20일 이동평균 계산
            volume_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
            
            # 현재 거래량과 비율 계산
            current_volume = volume[-1]
            ma_volume = volume_ma[-1]
            
            # 거래량 비율 계산 (0으로 나누기 방지)
            volume_ratio = current_volume / ma_volume if ma_volume > 0 else 1.0
            
            return {
                'current': float(current_volume),
                'ma': float(ma_volume),
                'ratio': float(volume_ratio),
                'trend': 'increasing' if volume_ratio > 1.2 else 'decreasing' if volume_ratio < 0.8 else 'neutral'
            }
        except Exception as e:
            self.logger.error(f"거래량 지표 계산 중 오류 발생: {e}")
            return {
                'current': 0.0,
                'ma': 0.0,
                'ratio': 1.0,
                'trend': 'neutral'
            }

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """변동성 계산"""
        try:
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연간화된 변동성
            return float(volatility)
        except Exception:
            return 0.02  # 기본값 2%

    def _prepare_market_data(self) -> pd.DataFrame:
        """시장 데이터 준비"""
        try:
            # CSV 파일에서 데이터 로드
            file_path = 'data/btcusdt_1m_2022_to_2025.csv'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
            
            df = pd.read_csv(file_path)
            
            # 날짜 컬럼을 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 1분 데이터를 5분으로 리샘플링
            df_resampled = pd.DataFrame()
            df_resampled['open'] = df['open'].resample('5min').first()
            df_resampled['high'] = df['high'].resample('5min').max()
            df_resampled['low'] = df['low'].resample('5min').min()
            df_resampled['close'] = df['close'].resample('5min').last()
            df_resampled['volume'] = df['volume'].resample('5min').sum()
            
            # NaN 값 처리
            df_resampled = df_resampled.ffill().bfill()
            
            # 인덱스를 컬럼으로 변환
            df_resampled.reset_index(inplace=True)
            
            # 랜덤한 시작점 선택 (마지막 2년 데이터는 제외)
            max_start_date = df_resampled['timestamp'].max() - pd.Timedelta(days=730)
            min_start_date = df_resampled['timestamp'].min()
            random_start = pd.Timestamp(np.random.uniform(min_start_date.value, max_start_date.value))
            
            # 선택된 시작점부터 2년 데이터 선택
            end_date = random_start + pd.Timedelta(days=730)
            df_resampled = df_resampled[(df_resampled['timestamp'] >= random_start) & 
                                      (df_resampled['timestamp'] <= end_date)]
            
            # 데이터 정렬
            df_resampled = df_resampled.sort_values('timestamp')
            
            # 기술적 지표 계산
            df_resampled = self._calculate_indicators(df_resampled)
            
            # NaN 값이 있는 행 제거
            df_resampled = df_resampled.dropna()
            
            self.logger.info(f"시뮬레이션 기간: {random_start.date()} ~ {end_date.date()}")
            self.logger.info(f"총 {len(df_resampled)}개의 데이터 포인트")
            
            return df_resampled
            
        except Exception as e:
            self.logger.error(f"시장 데이터 준비 중 오류 발생: {e}")
            raise

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # RSI 계산
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD 계산
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # 볼린저 밴드 계산
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # 스토캐스틱 계산
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ADX 계산
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            
            # 이동평균선 계산
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            
            # 변동성 계산 (ATR)
            df['volatility'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # 거래량 이동평균
            df['volume_sma'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
            
            # 거래량 비율 계산
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # NaN 값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 중 오류 발생: {e}")
            raise

    def _generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """트레이딩 시그널 생성"""
        try:
            indicators = market_data.get('indicators', {})
            if not indicators:
                return {'action': 'hold', 'reason': '충분한 지표 데이터 없음'}

            # 지표 값 가져오기
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            volatility = indicators.get('volatility', 0)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            current_price = market_data.get('current_price', 0)
            adx = indicators.get('adx', 25)

            # 매수 조건 점수 계산
            buy_score = 0
            buy_reasons = []
            
            # RSI 과매도 (더 엄격한 조건)
            if rsi < 30:
                buy_score += 2
                buy_reasons.append('RSI 과매도')
            elif rsi < 40:
                buy_score += 1
                buy_reasons.append('RSI 하락')
            
            # MACD 상승 (더 엄격한 조건)
            if macd > macd_signal and macd > 0:
                buy_score += 2
                buy_reasons.append('MACD 상승')
            elif macd > macd_signal:
                buy_score += 1
                buy_reasons.append('MACD 전환')
            
            # 스토캐스틱 (더 엄격한 조건)
            if stoch_k < 20 and stoch_k > stoch_d:
                buy_score += 2
                buy_reasons.append('스토캐스틱 과매도')
            elif stoch_k < 30 and stoch_k > stoch_d:
                buy_score += 1
                buy_reasons.append('스토캐스틱 상승')
            
            # 볼린저 밴드 (더 엄격한 조건)
            if current_price < bb_lower * 0.98:
                buy_score += 2
                buy_reasons.append('볼린저 밴드 하단')
            elif current_price < bb_lower:
                buy_score += 1
                buy_reasons.append('볼린저 밴드 근처')
            
            # 거래량 (더 엄격한 조건)
            if volume_ratio > 1.5:
                buy_score += 2
                buy_reasons.append('거래량 급증')
            elif volume_ratio > 1.2:
                buy_score += 1
                buy_reasons.append('거래량 증가')
            
            # ADX (추세 강도)
            if adx > 25:
                buy_score += 1
                buy_reasons.append('추세 강함')

            # 매도 조건 점수 계산
            sell_score = 0
            sell_reasons = []
            
            # RSI 과매수 (더 엄격한 조건)
            if rsi > 70:
                sell_score += 2
                sell_reasons.append('RSI 과매수')
            elif rsi > 60:
                sell_score += 1
                sell_reasons.append('RSI 상승')
            
            # MACD 하락 (더 엄격한 조건)
            if macd < macd_signal and macd < 0:
                sell_score += 2
                sell_reasons.append('MACD 하락')
            elif macd < macd_signal:
                sell_score += 1
                sell_reasons.append('MACD 전환')
            
            # 스토캐스틱 (더 엄격한 조건)
            if stoch_k > 80 and stoch_k < stoch_d:
                sell_score += 2
                sell_reasons.append('스토캐스틱 과매수')
            elif stoch_k > 70 and stoch_k < stoch_d:
                sell_score += 1
                sell_reasons.append('스토캐스틱 하락')
            
            # 볼린저 밴드 (더 엄격한 조건)
            if current_price > bb_upper * 1.02:
                sell_score += 2
                sell_reasons.append('볼린저 밴드 상단')
            elif current_price > bb_upper:
                sell_score += 1
                sell_reasons.append('볼린저 밴드 근처')
            
            # 거래량 (더 엄격한 조건)
            if volume_ratio > 1.5:
                sell_score += 2
                sell_reasons.append('거래량 급증')
            elif volume_ratio > 1.2:
                sell_score += 1
                sell_reasons.append('거래량 증가')
            
            # ADX (추세 강도)
            if adx > 25:
                sell_score += 1
                sell_reasons.append('추세 강함')

            # 신뢰도 계산 (더 엄격한 기준)
            confidence = max(buy_score, sell_score) / 10.0

            # 매매 신호 결정 (더 엄격한 조건)
            if buy_score >= 6:  # 최소 6개 이상의 조건 만족
                return {
                    'action': 'long',
                    'reason': ' + '.join(buy_reasons),
                    'confidence': confidence
                }
            elif sell_score >= 6:
                return {
                    'action': 'short',
                    'reason': ' + '.join(sell_reasons),
                    'confidence': confidence
                }

            return {'action': 'hold', 'reason': '조건 불충족', 'confidence': 0.0}

        except Exception as e:
            self.logger.error(f"시그널 생성 중 오류 발생: {e}")
            return {'action': 'hold', 'reason': f'오류 발생: {str(e)}', 'confidence': 0.0} 