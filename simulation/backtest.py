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
    def __init__(self, initial_capital: float = 500000.0):
        """백테스트 시뮬레이터 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 초기 설정
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_position = None
        self.trade_history = []
        self.daily_results = []
        
        # 리스크 매니저 초기화
        self.risk_manager = RiskManager()
        
        # 성과 지표
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        self.logger.info("백테스트 시뮬레이터 초기화 완료")
        
        # 컴포넌트 초기화
        self.exchange = Exchange()
        self.strategy = TradingStrategy()
        self.market_analyzer = MarketAnalyzer()
        
        # 로거 설정
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

    def _collect_market_data(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """시장 데이터 수집"""
        try:
            # 현재 데이터와 과거 데이터 준비
            current_row = df.iloc[current_idx]
            historical_data = df.iloc[max(0, current_idx - 20):current_idx + 1]
            
            # 기본 데이터
            market_data = {
                'timestamp': current_row.name,
                'open': float(current_row['open']),
                'high': float(current_row['high']),
                'low': float(current_row['low']),
                'close': float(current_row['close']),
                'volume': float(current_row['volume'])
            }
            
            # 기술적 지표
            if len(historical_data) > 0:
                market_data.update({
                    'rsi': float(current_row.get('rsi', 50)),
                    'macd': float(current_row.get('macd', 0)),
                    'macd_signal': float(current_row.get('macd_signal', 0)),
                    'bb_upper': float(current_row.get('bb_upper', market_data['close'])),
                    'bb_lower': float(current_row.get('bb_lower', market_data['close'])),
                    'stoch_k': float(current_row.get('stoch_k', 50)),
                    'stoch_d': float(current_row.get('stoch_d', 50)),
                    'adx': float(current_row.get('adx', 25)),
                    'di_plus': float(current_row.get('di_plus', 20)),
                    'di_minus': float(current_row.get('di_minus', 20)),
                    'volume_ratio': float(current_row.get('volume_ratio', 1.0))
                })
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 중 오류 발생: {e}")
            return None

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

    def _manage_positions(self, market_data: Dict, signal: Dict[str, Any]) -> None:
        """포지션 관리"""
        try:
            current_price = market_data['close']
            
            # 포지션 진입
            if not self.current_position and signal['action'] in ['buy', 'sell']:
                position_size, leverage = self.risk_manager.calculate_position_size(
                    market_data=market_data,
                    current_price=current_price
                )
                
                if position_size > 0:
                    self.current_position = {
                        'action': signal['action'],
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_time': market_data['timestamp'],
                        'leverage': leverage
                    }
                    self.logger.info(f"포지션 진입: {signal['action'].upper()} {position_size:.4f} @ {current_price:.2f} (레버리지: {leverage}x)")
            
            # 포지션 청산
            elif self.current_position:
                # 반대 신호로 인한 청산
                if signal['action'] != self.current_position['action'] and signal['action'] != 'none':
                    self._close_position(current_price, "반대 신호")
                    return
                
                # 손절/익절 조건 체크
                unrealized_pnl = (current_price - self.current_position['entry_price']) / self.current_position['entry_price']
                if self.current_position['action'] == 'sell':
                    unrealized_pnl = -unrealized_pnl
                
                # 손절 조건
                if unrealized_pnl < -0.02:  # 2% 손절
                    self._close_position(current_price, "손절")
                    return
                
                # 익절 조건
                if unrealized_pnl > 0.03:  # 3% 익절
                    self._close_position(current_price, "익절")
                    return
            
        except Exception as e:
            self.logger.error(f"포지션 관리 중 오류 발생: {str(e)}")
            raise

    def _execute_trade(self, market_data: Dict[str, Any], signal: Dict[str, Any]) -> None:
        """거래 실행"""
        try:
            current_price = market_data['close']
            
            # 포지션 관리
            self._manage_positions(market_data, signal)
            
            # 일일 결과 기록
            if self.current_position:
                pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
                self.daily_results.append({
                    'date': market_data['timestamp'],
                    'capital': self.current_capital + pnl,
                    'position': self.current_position['action'],
                    'size': self.current_position['size'],
                    'entry_price': self.current_position['entry_price'],
                    'current_price': current_price,
                    'pnl': pnl
                })
            else:
                self.daily_results.append({
                    'date': market_data['timestamp'],
                    'capital': self.current_capital,
                    'position': None,
                    'size': 0,
                    'entry_price': 0,
                    'current_price': current_price,
                    'pnl': 0
                })
                
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {e}")

    def _open_position(self, current_data: Dict, action: str, size: float) -> None:
        """포지션 진입"""
        try:
            current_price = current_data['close']
            
            # 포지션 정보 저장
            self.current_position = {
                'side': 'long' if action == 'buy' else 'short',
                'size': size,
                'entry_price': current_price,
                'entry_time': current_data['timestamp'],
                'initial_stop_loss': current_price * (0.98 if action == 'buy' else 1.02)  # 2% 손절
            }
            
            # 거래 기록
            trade = {
                'type': 'open',
                'side': self.current_position['side'],
                'price': current_price,
                'size': size,
                'timestamp': current_data['timestamp']
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"포지션 진입: {action.upper()} {size:.4f} @ {current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"포지션 진입 중 오류 발생: {e}")
            raise

    def _close_position(self, current_price: float, reason: str) -> None:
        """포지션 청산"""
        try:
            if not self.current_position:
                return
                
            # 수익/손실 계산
            entry_price = self.current_position['entry_price']
            position_size = self.current_position['size']
            pnl = (current_price - entry_price) * position_size
            
            if self.current_position['action'] == 'sell':
                pnl = -pnl
            
            # 거래 기록
            trade = {
                'type': 'close',
                'action': self.current_position['action'],
                'entry_price': entry_price,
                'exit_price': current_price,
                'size': position_size,
                'pnl': pnl,
                'reason': reason,
                'timestamp': self.current_position['entry_time']
            }
            self.trade_history.append(trade)
            
            # 자본금 업데이트
            self.current_capital += pnl
            
            # 최대 자본금 갱신
            self.peak_capital = max(self.peak_capital, self.current_capital)
            
            # 최대 낙폭 갱신
            if self.current_capital < self.peak_capital:
                drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # 승률 계산
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.logger.info(f"포지션 청산: {reason} - PNL: {pnl:.2f}")
            
            # 포지션 초기화
            self.current_position = None
            
        except Exception as e:
            self.logger.error(f"포지션 청산 중 오류 발생: {e}")
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
                start_date = pd.to_datetime(self.daily_results[0]['date'])
                end_date = pd.to_datetime(self.daily_results[-1]['date'])
                days = (end_date - start_date).days
                if days > 0:
                    annual_return = (Decimal('1') + total_return) ** (Decimal('365') / Decimal(str(days))) - Decimal('1')
                else:
                    annual_return = Decimal('0')
            else:
                annual_return = Decimal('0')
                
            # 승률 계산 수정 (type이 'close'인 거래만 계산)
            closed_trades = [t for t in self.trade_history if t.get('type') == 'close']
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

    def _generate_signal(self, current_data: Dict) -> Dict:
        """거래 신호 생성"""
        try:
            reasons = []
            buy_score = 0
            sell_score = 0
            
            # RSI 조건 (30/70 -> 35/65)
            if current_data.get('rsi') < 35:
                buy_score += 2
                reasons.append("RSI 과매도")
            elif current_data.get('rsi') > 65:
                sell_score += 2
                reasons.append("RSI 과매수")
                
            # MACD 조건
            if current_data.get('macd') > current_data.get('macd_signal', 0):
                buy_score += 1.5
                reasons.append("MACD 상승")
            else:
                sell_score += 1.5
                reasons.append("MACD 하락")
                
            # 볼린저 밴드 조건
            if current_data.get('close') < current_data.get('bb_lower', float('inf')):
                buy_score += 2
                reasons.append("BB 하단 돌파")
            elif current_data.get('close') > current_data.get('bb_upper', float('-inf')):
                sell_score += 2
                reasons.append("BB 상단 돌파")
                
            # 거래량 조건
            volume_ratio = current_data.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:  # 1.5 -> 1.2
                if current_data.get('close') > current_data.get('open', 0):
                    buy_score += 1.5
                    reasons.append("거래량 증가 + 상승")
                else:
                    sell_score += 1.5
                    reasons.append("거래량 증가 + 하락")
                    
            # ADX 조건 (25 -> 20)
            adx = current_data.get('adx', 0)
            if adx > 20:
                if current_data.get('di_plus', 0) > current_data.get('di_minus', 0):
                    buy_score += 1.5
                    reasons.append("ADX 강한 상승추세")
                else:
                    sell_score += 1.5
                    reasons.append("ADX 강한 하락추세")
                    
            # 스토캐스틱 조건
            if current_data.get('stoch_k', 0) < 30 and current_data.get('stoch_k', 0) > current_data.get('stoch_d', 0):
                buy_score += 1.5
                reasons.append("스토캐스틱 상승반전")
            elif current_data.get('stoch_k', 0) > 70 and current_data.get('stoch_k', 0) < current_data.get('stoch_d', 0):
                sell_score += 1.5
                reasons.append("스토캐스틱 하락반전")
                
            # 최소 점수 기준 완화 (6 -> 5)
            if buy_score >= 5:
                return {'action': 'buy', 'confidence': buy_score / 10, 'reasons': reasons}
            elif sell_score >= 5:
                return {'action': 'sell', 'confidence': sell_score / 10, 'reasons': reasons}
                
            return {'action': 'hold', 'confidence': 0, 'reasons': reasons}
            
        except Exception as e:
            self.logger.error(f"거래 신호 생성 중 오류 발생: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasons': ["오류 발생"]}

    def _run_simulation(self) -> None:
        """시뮬레이션 실행"""
        try:
            self.logger.info("시뮬레이션 시작")
            
            # 시장 데이터 준비
            market_data = self._prepare_market_data()
            if market_data is None:
                self.logger.error("시장 데이터 준비 실패")
                return
                
            # 시뮬레이션 기간 설정
            start_date = market_data.index[0]
            end_date = market_data.index[-1]
            self.logger.info(f"시뮬레이션 기간: {start_date} ~ {end_date}")
            
            # 초기화
            self.current_capital = self.initial_capital
            self.current_position = None
            self.daily_results = []
            self.trade_history = []
            
            # 시뮬레이션 실행
            for idx, row in market_data.iterrows():
                # 현재 시장 데이터
                current_data = row.to_dict()
                current_data['name'] = idx
                
                # 거래 신호 생성
                signal = self._generate_signal(current_data)
                
                # 거래 실행
                self._execute_trade(current_data, signal)
                
            # 결과 저장
            self._save_results()
            
            self.logger.info("시뮬레이션 완료")
            
        except Exception as e:
            self.logger.error(f"시뮬레이션 실행 중 오류 발생: {e}")

    def _save_results(self):
        """결과 저장"""
        try:
            # 결과 저장 로직
            pass
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {e}")
            raise 