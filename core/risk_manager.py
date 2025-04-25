from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import settings
import logging
from utils.logger import setup_logger
from .exchange import Exchange
from decimal import Decimal

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.risk_level = 'medium'  # 기본 리스크 레벨
        self.max_drawdown = 0.5  # 최대 드로다운 50%
        self.daily_loss_limit = 0.1  # 일일 손실 한도 10%
        self.position_size = 1.0  # 풀시드 포지션
        self.leverage = 50  # 50배 레버리지
        self.min_leverage = 30  # 최소 레버리지
        self.max_leverage = 50  # 최대 레버리지
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5  # 연속 손실 제한
        self.trade_history = []
        self.volatility_threshold = 0.02  # 변동성 임계값
        self.risk_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        self.initial_capital = Decimal('10000.0')  # 기본 초기 자본금
        self.max_daily_loss = 0.1  # 일일 최대 손실 한도 (10%)
        self.max_position_size = 1.0  # 최대 포지션 크기 (100%)
        self.max_daily_trades = 5     # 일일 최대 거래 횟수
        self.consecutive_loss_limit = 3  # 연속 손실 제한
        
        self.daily_pnl = 0.0        # 일일 손익
        self.total_pnl = 0.0        # 총 손익
        self.max_equity = 0.0       # 최대 자산
        self.current_equity = self.initial_capital   # 현재 자산
        self.last_reset_time = None # 마지막 리셋 시간
        
        logger.info("리스크 매니저 초기화 완료")
        
    def _validate_account_state(self) -> bool:
        """계좌 상태 검증"""
        try:
            if self.current_equity <= 0:
                logger.warning("계좌 자산이 0 이하입니다. 초기 자본금으로 재설정합니다.")
                self.current_equity = self.initial_capital
                self.max_equity = self.initial_capital
                self.daily_pnl = 0.0
                self.total_pnl = 0.0
                self.consecutive_losses = 0
                self.risk_level = 'medium'
                return True
                
            if self.max_equity <= 0:
                self.max_equity = self.current_equity
                
            return True
            
        except Exception as e:
            logger.error(f"계좌 상태 검증 중 오류 발생: {e}")
            return False

    def update_account_state(self, equity: float, pnl: float) -> None:
        """계좌 상태 업데이트"""
        try:
            # 일일 손익 리셋 체크
            current_time = datetime.now()
            if (self.last_reset_time is None or 
                (current_time - self.last_reset_time).days >= 1):
                self.daily_pnl = Decimal('0.0')
                self.last_reset_time = current_time
                
            # 계좌 상태 업데이트
            self.current_equity = Decimal(str(equity))
            self.daily_pnl += Decimal(str(pnl))
            self.total_pnl += Decimal(str(pnl))
            
            # 최대 자산 업데이트
            if self.current_equity > self.max_equity:
                self.max_equity = self.current_equity
                
            # 리스크 레벨 업데이트
            self._update_risk_level()
            
        except Exception as e:
            logger.error(f"계좌 상태 업데이트 중 오류 발생: {e}")

    def _update_risk_level(self) -> None:
        """리스크 레벨 업데이트"""
        try:
            # 일일 손실 체크
            daily_loss_limit = Decimal(str(self.daily_loss_limit)) * self.max_equity
            if self.daily_pnl <= -daily_loss_limit:
                self.risk_level = 'very_high'
                return
                
            # 드로다운 체크
            drawdown = (self.max_equity - self.current_equity) / self.max_equity
            if drawdown >= Decimal(str(self.max_drawdown)):
                self.risk_level = 'very_high'
                return
                
            # 연속 손실 체크
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.risk_level = 'high'
                return
                
            # 기본 리스크 레벨
            self.risk_level = 'medium'
            
        except Exception as e:
            logger.error(f"리스크 레벨 업데이트 중 오류 발생: {e}")
            self.risk_level = 'high'

    def calculate_position_size(self, market_data: Dict[str, Any], current_price: float) -> Tuple[float, int]:
        """포지션 크기 계산"""
        try:
            # 기본 포지션 크기 (계좌의 15%)
            base_size = self.current_equity * Decimal('0.15')
            
            # 변동성 기반 조정
            volatility = market_data.get('volatility', 0.02)
            if volatility > 0.03:  # 높은 변동성
                base_size *= Decimal('0.8')
            elif volatility < 0.01:  # 낮은 변동성
                base_size *= Decimal('1.5')
                
            # 리스크 레벨 기반 조정
            if self.risk_level == 'high':
                base_size *= Decimal('0.8')
            elif self.risk_level == 'low':
                base_size *= Decimal('1.5')
                
            # 포지션 크기 제한
            position_size = max(min(base_size / Decimal(str(current_price)), 
                                 Decimal(str(self.max_position_size))), 
                             Decimal('0.05'))
            
            # 레버리지 계산
            leverage = self._calculate_leverage(volatility)
            
            return float(position_size), leverage
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류 발생: {e}")
            return 0.15, 20  # 기본값 반환

    def _calculate_leverage(self, volatility: float) -> int:
        """레버리지 계산"""
        try:
            # 기본 레버리지 증가
            base_leverage = self.min_leverage
            
            # 변동성 기반 조정
            if volatility > 0.03:  # 높은 변동성
                base_leverage = max(self.min_leverage, base_leverage * 0.8)
            elif volatility < 0.01:  # 낮은 변동성
                base_leverage = min(self.max_leverage, base_leverage * 2.0)
                
            # 리스크 레벨 기반 조정
            if self.risk_level == 'very_high':
                base_leverage = max(self.min_leverage, base_leverage * 0.8)
            elif self.risk_level == 'high':
                base_leverage = min(self.max_leverage, base_leverage * 1.2)
            elif self.risk_level == 'low':
                base_leverage = min(self.max_leverage, base_leverage * 1.5)
                
            # 레버리지 제한
            leverage = max(min(base_leverage, self.max_leverage), self.min_leverage)
            
            return int(leverage)
            
        except Exception as e:
            logger.error(f"레버리지 계산 중 오류 발생: {e}")
            return 20  # 기본값 반환

    def check_risk_limits(self, market_data: Dict[str, Any] = None) -> bool:
        """리스크 한도 체크"""
        try:
            if not self._validate_account_state():
                return False
                
            # 일일 손실 한도 체크
            daily_loss_limit = Decimal(str(self.daily_loss_limit)) * self.max_equity
            if self.daily_pnl <= -daily_loss_limit:
                logger.warning(f"일일 손실 한도 초과: {self.daily_pnl:.2%}")
                return False
                
            # 드로다운 한도 체크
            drawdown = (self.max_equity - self.current_equity) / self.max_equity
            if drawdown >= Decimal(str(self.max_drawdown)):
                logger.warning(f"드로다운 한도 초과: {drawdown:.2%}")
                return False
                
            # 연속 손실 한도 체크
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning(f"연속 손실 한도 초과: {self.consecutive_losses}회")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"리스크 한도 체크 중 오류 발생: {e}")
            return False

    def update_trade_result(self, is_profit: bool, pnl: float) -> None:
        """거래 결과 업데이트"""
        try:
            if is_profit:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                
            self.update_account_state(self.current_equity, pnl)
            
        except Exception as e:
            logger.error(f"거래 결과 업데이트 중 오류 발생: {e}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """리스크 메트릭 반환"""
        try:
            drawdown = (self.max_equity - self.current_equity) / self.max_equity if self.max_equity > 0 else 0.0
            
            return {
                'daily_pnl': float(self.daily_pnl),
                'total_pnl': float(self.total_pnl),
                'current_equity': float(self.current_equity),
                'max_equity': float(self.max_equity),
                'drawdown': float(drawdown),
                'consecutive_losses': self.consecutive_losses,
                'risk_level': self.risk_level
            }
            
        except Exception as e:
            logger.error(f"리스크 메트릭 계산 중 오류 발생: {e}")
            return {}

    def reset(self) -> None:
        """리스크 관리자 초기화"""
        try:
            self.daily_pnl = 0.0
            self.total_pnl = 0.0
            self.max_equity = self.current_equity
            self.last_reset_time = datetime.now()
            self.consecutive_losses = 0
            self.risk_level = 'medium'
            
        except Exception as e:
            logger.error(f"리스크 관리자 초기화 중 오류 발생: {e}")

    def check_liquidation_risk(self, side: str, entry_price: float, current_price: float) -> bool:
        """청산 위험 체크"""
        try:
            liquidation_price = self.exchange.calculate_liquidation_price(
                side, entry_price, settings.LEVERAGE
            )
            
            if side == 'long':
                risk = (current_price - liquidation_price) / current_price
            else:
                risk = (liquidation_price - current_price) / current_price
                
            if risk < 0.05:  # 청산가까지 5% 이내
                logger.warning(f"청산 위험 높음: 현재가격 {current_price}, 청산가격 {liquidation_price}")
                return False
            return True
            
        except Exception as e:
            logger.error(f"청산 위험 체크 실패: {e}")
            return False

    def update_pnl(self, pnl: float, trade_info: Dict[str, Any]):
        """손익 업데이트"""
        self.daily_pnl += pnl
        self.trade_history.append(trade_info)
        
        # 최대 드로다운 업데이트
        balance = self.exchange.get_balance()['total']
        if balance > self.peak_balance:
            self.peak_balance = balance
        else:
            drawdown = (self.peak_balance - balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def calculate_win_rate(self) -> float:
        """승률 계산"""
        if not self.trade_history:
            return 0.0
            
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        return winning_trades / len(self.trade_history)

    def can_trade(self, current_capital: float, trade_history: List[Dict]) -> bool:
        """거래 가능 여부 확인"""
        try:
            # 기본 계좌 상태 검증
            if not self._validate_account_state():
                return False

            # 일일 거래 횟수 제한 확인
            current_time = datetime.now()
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            today_trades = [trade for trade in trade_history 
                          if isinstance(trade.get('timestamp'), datetime) and 
                          trade['timestamp'] >= today_start]
            
            if len(today_trades) >= self.max_daily_trades:
                logger.warning(f"일일 최대 거래 횟수 초과: {len(today_trades)}/{self.max_daily_trades}")
                return False

            # 연속 손실 제한 확인
            if self.consecutive_losses >= self.consecutive_loss_limit:
                logger.warning(f"연속 손실 제한 초과: {self.consecutive_losses}/{self.consecutive_loss_limit}")
                return False

            # 최소 자본금 확인
            min_capital = self.initial_capital * Decimal('0.5')  # 초기 자본금의 50%
            if Decimal(str(current_capital)) < min_capital:
                logger.warning(f"최소 자본금 미달: {current_capital}/{float(min_capital)}")
                return False

            return True

        except Exception as e:
            logger.error(f"거래 가능 여부 확인 중 오류 발생: {str(e)}")
            return False 