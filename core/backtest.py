import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from config.settings import settings
from utils.logger import setup_logger
from .strategy import TradingStrategy

logger = setup_logger(__name__)

class BacktestEngine:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.initial_balance = 10000  # 초기 자본 10,000 USDT
        self.slippage_pct = 0.0005  # 0.05% 슬리피지
        self.fee_rate = 0.0004  # 0.04% 수수료

    def apply_slippage(self, price: float, side: str) -> float:
        """슬리피지 적용"""
        if side == 'buy':
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def calculate_fee(self, amount: float, price: float) -> float:
        """수수료 계산"""
        return amount * price * self.fee_rate

    def run(self, data1: pd.DataFrame, data5: pd.DataFrame) -> Dict[str, Any]:
        """백테스트 실행"""
        try:
            current_balance = self.initial_balance
            position = None
            entry_price = 0
            entry_time = None
            max_balance = self.initial_balance
            max_drawdown = 0
            
            for i in range(len(data1)):
                current_data1 = data1.iloc[:i+1]
                current_data5 = data5.iloc[:i+1]
                current_price = data1['close'].iloc[i]
                current_time = data1.index[i]
                
                # 포지션이 없는 경우
                if position is None:
                    # 진입 조건 체크
                    side = self.strategy.check_entry_conditions(current_data1, current_data5)
                    if side:
                        # 포지션 사이즈 계산
                        position_size = self.strategy.risk_manager.calculate_position_size(current_price)
                        if side == 'aggressive_long':
                            position_size = current_balance / current_price
                        
                        if position_size > 0:
                            # 슬리피지 적용
                            entry_price = self.apply_slippage(current_price, 'buy' if 'long' in side else 'sell')
                            
                            # 수수료 차감
                            fee = self.calculate_fee(position_size, entry_price)
                            current_balance -= fee
                            
                            position = side
                            entry_time = current_time
                            self.trades.append({
                                'type': 'entry',
                                'side': side,
                                'price': entry_price,
                                'time': current_time,
                                'size': position_size,
                                'fee': fee
                            })
                
                # 포지션이 있는 경우
                else:
                    # Crash/Spike 감지
                    if self.strategy.detect_crash(current_data1, current_data5):
                        # 포지션 청산
                        exit_price = self.apply_slippage(current_price, 'sell' if position == 'long' else 'buy')
                        fee = self.calculate_fee(position_size, exit_price)
                        
                        if position == 'long':
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
                        current_balance *= (1 + pnl)
                        current_balance -= fee
                        
                        self.trades.append({
                            'type': 'exit',
                            'side': position,
                            'price': exit_price,
                            'time': current_time,
                            'pnl': pnl,
                            'fee': fee,
                            'reason': 'crash'
                        })
                        
                        # 숏 진입
                        short_size = self.strategy.risk_manager.calculate_position_size(current_price)
                        entry_price = self.apply_slippage(current_price, 'sell')
                        fee = self.calculate_fee(short_size, entry_price)
                        current_balance -= fee
                        
                        position = 'short'
                        entry_time = current_time
                        self.trades.append({
                            'type': 'entry',
                            'side': 'short',
                            'price': entry_price,
                            'time': current_time,
                            'size': short_size,
                            'fee': fee,
                            'reason': 'crash_short'
                        })
                        continue
                        
                    if self.strategy.detect_spike(current_data1):
                        # 포지션 청산
                        exit_price = self.apply_slippage(current_price, 'sell' if position == 'long' else 'buy')
                        fee = self.calculate_fee(position_size, exit_price)
                        
                        if position == 'long':
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
                        current_balance *= (1 + pnl)
                        current_balance -= fee
                        
                        self.trades.append({
                            'type': 'exit',
                            'side': position,
                            'price': exit_price,
                            'time': current_time,
                            'pnl': pnl,
                            'fee': fee,
                            'reason': 'spike'
                        })
                        
                        position = None
                        entry_price = 0
                        entry_time = None
                        continue
                    
                    # 청산 조건 체크
                    should_exit, close_pct = self.strategy.check_exit_conditions(current_data1, position, entry_price)
                    if should_exit:
                        # 포지션 청산
                        exit_price = self.apply_slippage(current_price, 'sell' if position == 'long' else 'buy')
                        close_amount = position_size * close_pct
                        fee = self.calculate_fee(close_amount, exit_price)
                        
                        if position == 'long':
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
                        current_balance *= (1 + pnl * close_pct)
                        current_balance -= fee
                        
                        self.trades.append({
                            'type': 'exit',
                            'side': position,
                            'price': exit_price,
                            'time': current_time,
                            'pnl': pnl,
                            'fee': fee,
                            'close_pct': close_pct
                        })
                        
                        if close_pct == 1.0:
                            position = None
                            entry_price = 0
                            entry_time = None
                        else:
                            position_size *= (1 - close_pct)
                
                # 자본 곡선 업데이트
                self.equity_curve.append(current_balance)
                
                # 최대 드로다운 계산
                if current_balance > max_balance:
                    max_balance = current_balance
                else:
                    drawdown = (max_balance - current_balance) / max_balance
                    max_drawdown = max(max_drawdown, drawdown)
            
            # 성과 지표 계산
            metrics = self.calculate_metrics()
            metrics['max_drawdown'] = max_drawdown
            
            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {e}")
            return {}

    def calculate_metrics(self) -> Dict[str, float]:
        """성과 지표 계산"""
        try:
            # 수익률
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            total_return = (self.equity_curve[-1] / self.initial_balance) - 1
            
            # 샤프비율
            risk_free_rate = 0.02  # 연 2% 무위험 수익률 가정
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 승률
            winning_trades = len([t for t in self.trades if t['type'] == 'exit' and t['pnl'] > 0])
            total_trades = len([t for t in self.trades if t['type'] == 'exit'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 평균 수익/손실
            profits = [t['pnl'] for t in self.trades if t['type'] == 'exit' and t['pnl'] > 0]
            losses = [t['pnl'] for t in self.trades if t['type'] == 'exit' and t['pnl'] < 0]
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # 수익 요인
            profit_factors = {
                'crash': len([t for t in self.trades if t.get('reason') == 'crash' and t['pnl'] > 0]),
                'spike': len([t for t in self.trades if t.get('reason') == 'spike' and t['pnl'] > 0]),
                'normal': len([t for t in self.trades if t['type'] == 'exit' and t['pnl'] > 0 and not t.get('reason')])
            }
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_trades': total_trades,
                'profit_factors': profit_factors
            }
            
        except Exception as e:
            logger.error(f"성과 지표 계산 중 오류 발생: {e}")
            return {} 