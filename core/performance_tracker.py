import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.max_drawdown: float = 0.0
        self.current_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.win_rate: float = 0.0
        self.profit_factor: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.sortino_ratio: float = 0.0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0
        self.last_update_time: Optional[datetime] = None
        self.initial_balance: float = 0.0
        self.market_state_performance: Dict[str, Dict[str, float]] = {}
        
    def add_trade(self, trade_info: Dict[str, Any]):
        """거래 정보 추가"""
        try:
            self.trades.append(trade_info)
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"거래 정보 추가 중 오류 발생: {e}")
            
    def update_equity(self, equity: float):
        """자본 곡선 업데이트"""
        try:
            self.equity_curve.append(equity)
            
            # 최대 자본 업데이트
            if equity > self.peak_equity:
                self.peak_equity = equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
                
        except Exception as e:
            logger.error(f"자본 곡선 업데이트 중 오류 발생: {e}")
            
    def calculate_daily_returns(self):
        """일일 수익률 계산"""
        try:
            if len(self.equity_curve) < 2:
                return
                
            # 일일 수익률 계산
            daily_changes = pd.Series(self.equity_curve).pct_change().dropna()
            self.daily_returns = daily_changes.tolist()
            
        except Exception as e:
            logger.error(f"일일 수익률 계산 중 오류 발생: {e}")
            
    def calculate_risk_metrics(self):
        """리스크 지표 계산"""
        try:
            if not self.daily_returns:
                self.calculate_daily_returns()
                
            returns = np.array(self.daily_returns)
            
            # 샤프 비율 계산
            risk_free_rate = 0.02 / 252  # 연 2% 무위험 수익률 가정
            excess_returns = returns - risk_free_rate
            self.sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            
            # 소르티노 비율 계산
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                self.sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            else:
                self.sortino_ratio = np.inf
                
        except Exception as e:
            logger.error(f"리스크 지표 계산 중 오류 발생: {e}")
            
    def _update_metrics(self):
        """성과 지표 업데이트"""
        try:
            if not self.trades:
                return
                
            # 승률 계산
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            self.win_rate = len(winning_trades) / len(self.trades)
            
            # 수익 요인 계산
            total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
            self.profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
            
            # 연속 승/패 계산
            current_streak = 0
            max_streak = 0
            last_pnl = None
            
            for trade in self.trades:
                if last_pnl is None:
                    last_pnl = trade['pnl']
                    current_streak = 1
                    continue
                    
                if (trade['pnl'] > 0 and last_pnl > 0) or (trade['pnl'] < 0 and last_pnl < 0):
                    current_streak += 1
                else:
                    current_streak = 1
                    
                max_streak = max(max_streak, current_streak)
                last_pnl = trade['pnl']
                
            if self.trades[-1]['pnl'] > 0:
                self.consecutive_wins = current_streak
                self.max_consecutive_wins = max(self.max_consecutive_wins, current_streak)
            else:
                self.consecutive_losses = current_streak
                self.max_consecutive_losses = max(self.max_consecutive_losses, current_streak)
                
        except Exception as e:
            logger.error(f"성과 지표 업데이트 중 오류 발생: {e}")
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성과 지표 반환"""
        try:
            return {
                'total_trades': len(self.trades),
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses,
                'current_equity': self.equity_curve[-1] if self.equity_curve else 0.0,
                'peak_equity': self.peak_equity
            }
            
        except Exception as e:
            logger.error(f"성과 지표 반환 중 오류 발생: {e}")
            return {} 

    def update_metrics(self, current_balance: float, position: Optional[Dict[str, Any]] = None, 
                      market_state: Optional[str] = None) -> None:
        """성과 지표 업데이트"""
        try:
            # 현재 시간
            current_time = datetime.now()
            
            # 일일 수익률 계산
            if self.last_update_time and current_time.date() > self.last_update_time.date():
                self.daily_returns = []
                self.last_update_time = current_time
                
            # 현재 수익률 계산
            if self.initial_balance > 0:
                current_return = (current_balance - self.initial_balance) / self.initial_balance
                self.daily_returns.append(current_return)
                
            # 최대 드로다운 업데이트
            if current_balance > self.peak_equity:
                self.peak_equity = current_balance
            else:
                current_drawdown = (self.peak_equity - current_balance) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
            # 포지션 정보가 있는 경우 추가 업데이트
            if position:
                self._update_position_metrics(position)
                
            # 시장 상태가 있는 경우 추가 업데이트
            if market_state:
                self._update_market_metrics(market_state)
                
            self.last_update_time = current_time
            
        except Exception as e:
            logger.error(f"성과 지표 업데이트 중 오류 발생: {e}")
            
    def _update_position_metrics(self, position: Dict[str, Any]) -> None:
        """포지션 관련 지표 업데이트"""
        try:
            # 연속 승/패 업데이트
            if position.get('pnl', 0) > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
        except Exception as e:
            logger.error(f"포지션 지표 업데이트 중 오류 발생: {e}")
            
    def _update_market_metrics(self, market_state: str) -> None:
        """시장 상태 관련 지표 업데이트"""
        try:
            # 시장 상태별 성과 추적
            if market_state not in self.market_state_performance:
                self.market_state_performance[market_state] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0
                }
                
        except Exception as e:
            logger.error(f"시장 상태 지표 업데이트 중 오류 발생: {e}") 