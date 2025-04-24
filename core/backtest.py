import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from .market_analyzer import MarketAnalyzer
from .strategy import TradingStrategy
from .position_manager import PositionManager
from .performance_tracker import PerformanceTracker
from .risk_manager import RiskManager
from .utils import calculate_slippage, calculate_commission

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_balance: float = 500000):
        self.initial_balance = initial_balance
        self.market_analyzer = MarketAnalyzer()
        self.strategy = TradingStrategy()
        self.position_manager = PositionManager(initial_balance)
        self.performance_tracker = PerformanceTracker()
        self.risk_manager = RiskManager()
        
        # 백테스팅 설정
        self.slippage_model = 'normal'  # normal, aggressive, conservative
        self.commission_rate = 0.0004  # 0.04%
        self.min_trade_size = 0.001  # 최소 거래 수량
        self.partial_liquidation = True  # 부분 청산 사용 여부
        self.max_partial_liquidation = 0.5  # 최대 부분 청산 비율
        
    async def run(self, data: pd.DataFrame, market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """백테스팅 실행"""
        try:
            results = {
                'trades': [],
                'equity_curve': [],
                'metrics': {},
                'market_conditions': market_conditions or {}
            }
            
            current_balance = self.initial_balance
            current_position = None
            
            for i in range(len(data)):
                try:
                    # 현재 시장 데이터
                    current_data = data.iloc[i]
                    
                    # 시장 분석
                    market_analysis = self.market_analyzer.analyze_market_condition(current_data)
                    
                    # 리스크 관리
                    if not self.risk_manager.check_risk_limits(market_analysis):
                        continue
                        
                    # 매매 신호 생성
                    signal = self.strategy.generate_signal(market_analysis)
                    
                    if signal:
                        # 슬리피지 계산
                        slippage = calculate_slippage(
                            current_data['volume'],
                            signal['amount'],
                            self.slippage_model
                        )
                        
                        # 커미션 계산
                        commission = calculate_commission(
                            signal['amount'],
                            current_data['close'],
                            self.commission_rate
                        )
                        
                        # 거래 실행
                        trade_result = await self._execute_trade(
                            signal,
                            current_data,
                            slippage,
                            commission
                        )
                        
                        if trade_result:
                            results['trades'].append(trade_result)
                            current_balance = trade_result['balance_after']
                            current_position = trade_result['position']
                            
                    # 자본 곡선 업데이트
                    results['equity_curve'].append(current_balance)
                    
                    # 성과 지표 업데이트
                    if i % 1440 == 0:  # 매일
                        daily_metrics = self.performance_tracker.calculate_daily_metrics(results['trades'])
                        results['metrics'].update(daily_metrics)
                        
                except Exception as e:
                    logger.error(f"백테스팅 중 오류 발생: {e}")
                    continue
                    
            # 최종 성과 지표 계산
            final_metrics = self._calculate_final_metrics(results)
            results['metrics'].update(final_metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"백테스팅 실행 중 오류 발생: {e}")
            return None
            
    async def _execute_trade(self, signal: Dict[str, Any], data: pd.DataFrame,
                           slippage: float, commission: float) -> Optional[Dict[str, Any]]:
        """거래 실행 시뮬레이션"""
        try:
            if signal['type'] == 'buy':
                # 포지션 크기 계산
                position_size = self.position_manager.calculate_position_size(
                    data,
                    signal['risk_level']
                )
                
                if position_size < self.min_trade_size:
                    return None
                    
                # 레버리지 계산
                leverage = self.position_manager.calculate_leverage(data)
                
                # 거래 실행
                entry_price = data['close'] * (1 + slippage)
                position_value = position_size * entry_price
                commission_paid = position_value * commission
                
                # 포지션 정보 업데이트
                self.position_manager.update_position(
                    entry_price,
                    position_size,
                    leverage,
                    data
                )
                
                return {
                    'type': 'buy',
                    'time': data.name,
                    'price': entry_price,
                    'amount': position_size,
                    'leverage': leverage,
                    'commission': commission_paid,
                    'slippage': slippage,
                    'position': 'long',
                    'balance_after': self.position_manager.current_balance - commission_paid
                }
                
            elif signal['type'] == 'sell':
                if not self.position_manager.position_size:
                    return None
                    
                # 부분 청산 여부 확인
                if self.partial_liquidation:
                    liquidation_ratio = min(
                        self.max_partial_liquidation,
                        self.risk_manager.calculate_liquidation_ratio(data)
                    )
                    position_size = self.position_manager.position_size * liquidation_ratio
                else:
                    position_size = self.position_manager.position_size
                    
                # 거래 실행
                exit_price = data['close'] * (1 - slippage)
                position_value = position_size * exit_price
                commission_paid = position_value * commission
                
                # PnL 계산
                pnl = self.position_manager.update_pnl(exit_price)
                
                return {
                    'type': 'sell',
                    'time': data.name,
                    'price': exit_price,
                    'amount': position_size,
                    'leverage': self.position_manager.leverage,
                    'commission': commission_paid,
                    'slippage': slippage,
                    'pnl': pnl,
                    'position': 'none' if position_size == self.position_manager.position_size else 'partial',
                    'balance_after': self.position_manager.current_balance - commission_paid
                }
                
        except Exception as e:
            logger.error(f"거래 실행 시뮬레이션 중 오류 발생: {e}")
            return None
            
    def _calculate_final_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """최종 성과 지표 계산"""
        try:
            trades = results['trades']
            equity_curve = results['equity_curve']
            
            # 기본 지표
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 수익률
            total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance
            annualized_return = (1 + total_return) ** (365 / len(equity_curve)) - 1
            
            # 리스크 지표
            returns = pd.Series(equity_curve).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
            
            # 드로다운
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # 거래 통계
            avg_trade_pnl = np.mean([t.get('pnl', 0) for t in trades]) if trades else 0
            avg_winning_trade = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
            avg_losing_trade = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade_pnl': avg_trade_pnl,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade
            }
            
        except Exception as e:
            logger.error(f"최종 성과 지표 계산 중 오류 발생: {e}")
            return {} 