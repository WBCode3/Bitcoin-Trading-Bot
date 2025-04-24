import numpy as np
from typing import Dict, Any, Optional
import logging
from decimal import Decimal, ROUND_DOWN
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_position = None
        self.position_history = []
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.leverage = 1.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.daily_loss_limit = 0.02  # 일일 손실 한도 2%
        self.daily_pnl = 0.0
        
        logger.info(f"포지션 매니저 초기화 완료 (초기 자본: {initial_balance:,.2f} USDT)")
        
    def calculate_position_size(self, market_analysis: Dict[str, Any], 
                              risk_level: str) -> float:
        """시장 상황과 리스크 레벨에 따른 포지션 크기 계산"""
        try:
            base_size = self.current_balance * 0.05  # 기본 포지션 크기 5%
            
            # 변동성 기반 조정
            volatility_factor = self._calculate_volatility_factor(
                market_analysis['volatility']
            )
            
            # 리스크 레벨 기반 조정
            risk_factor = self._calculate_risk_factor(risk_level)
            
            # 최종 포지션 크기 계산
            position_size = base_size * volatility_factor * risk_factor
            
            # 최대 포지션 크기 제한
            max_position = self.current_balance * 0.1  # 최대 10%
            return min(position_size, max_position)
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류 발생: {e}")
            return 0.0
            
    def calculate_leverage(self, market_analysis: Dict[str, Any]) -> float:
        """시장 상황에 따른 레버리지 계산"""
        try:
            base_leverage = 5.0  # 기본 레버리지
            
            # 변동성 기반 조정
            if market_analysis['volatility'] > 0.03:
                leverage_factor = 0.5
            elif market_analysis['volatility'] > 0.02:
                leverage_factor = 0.7
            elif market_analysis['volatility'] < 0.01:
                leverage_factor = 1.2
            else:
                leverage_factor = 1.0
                
            # 최종 레버리지 계산
            leverage = base_leverage * leverage_factor
            
            # 최대 레버리지 제한
            return min(leverage, 10.0)
            
        except Exception as e:
            logger.error(f"레버리지 계산 중 오류 발생: {e}")
            return 1.0
            
    def calculate_stop_loss(self, entry_price: float, 
                          market_analysis: Dict[str, Any]) -> float:
        """진입가격과 시장 상황에 따른 손절가 계산"""
        try:
            # 기본 손절폭
            base_stop_loss = 0.015  # 1.5%
            
            # 변동성 기반 조정
            volatility_factor = market_analysis['volatility'] / 0.02
            
            # 최종 손절폭 계산
            stop_loss_pct = base_stop_loss * volatility_factor
            
            # 최대 손절폭 제한
            stop_loss_pct = min(stop_loss_pct, 0.03)  # 최대 3%
            
            return entry_price * (1 - stop_loss_pct)
            
        except Exception as e:
            logger.error(f"손절가 계산 중 오류 발생: {e}")
            return entry_price * 0.98
            
    def calculate_take_profit(self, entry_price: float, 
                            stop_loss: float) -> float:
        """진입가격과 손절가에 따른 익절가 계산"""
        try:
            # 리스크:리워드 비율 (1:2)
            risk = entry_price - stop_loss
            return entry_price + (risk * 2)
            
        except Exception as e:
            logger.error(f"익절가 계산 중 오류 발생: {e}")
            return entry_price * 1.03
            
    def update_position(self, entry_price: float, size: float, leverage: float, 
                       market_analysis: Dict[str, Any]) -> None:
        """포지션 정보 업데이트"""
        try:
            # 포지션 크기 계산
            position_value = size * entry_price
            
            # 포지션 정보 업데이트
            self.current_position = {
                'entry_price': entry_price,
                'size': size,
                'leverage': leverage,
                'value': position_value,
                'entry_time': datetime.now(),
                'market_state': market_analysis.get('market_state', 'normal'),
                'risk_level': market_analysis.get('risk_level', 'medium')
            }
            
            # 포지션 히스토리 추가
            self.position_history.append(self.current_position.copy())
            
            logger.info(f"포지션 업데이트: {self.current_position}")
            
        except Exception as e:
            logger.error(f"포지션 업데이트 중 오류 발생: {e}")
            
    def close_position(self, exit_price: float) -> Dict[str, Any]:
        """포지션 청산"""
        try:
            if not self.current_position:
                logger.warning("청산할 포지션이 없습니다.")
                return None
                
            # 손익 계산
            entry_price = self.current_position['entry_price']
            size = self.current_position['size']
            leverage = self.current_position['leverage']
            
            pnl = (exit_price - entry_price) * size * leverage
            pnl_percentage = pnl / self.current_position['value']
            
            # 포지션 정보 업데이트
            closed_position = self.current_position.copy()
            closed_position.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'pnl': pnl,
                'pnl_percentage': pnl_percentage
            })
            
            # 현재 포지션 초기화
            self.current_position = None
            
            # 잔고 업데이트
            self.current_balance += pnl
            self.peak_balance = max(self.peak_balance, self.current_balance)
            
            # 최대 드로다운 업데이트
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            logger.info(f"포지션 청산: {closed_position}")
            return closed_position
            
        except Exception as e:
            logger.error(f"포지션 청산 중 오류 발생: {e}")
            return None
            
    def get_position_info(self) -> Dict[str, Any]:
        """현재 포지션 정보 반환"""
        if not self.current_position:
            return None
            
        return self.current_position.copy()
            
    def update_pnl(self, current_price: float) -> float:
        """현재 가격 기준 PnL 계산 및 업데이트"""
        try:
            if self.position_size == 0:
                return 0.0
                
            pnl = (current_price - self.entry_price) * self.position_size * self.leverage
            self.daily_pnl += pnl
            
            # 최대 드로다운 업데이트
            if pnl < 0:
                drawdown = abs(pnl) / self.current_balance
                self.max_drawdown = max(self.max_drawdown, drawdown)
                
            return pnl
            
        except Exception as e:
            logger.error(f"PnL 업데이트 중 오류 발생: {e}")
            return 0.0
            
    def check_risk_limits(self) -> bool:
        """리스크 한도 체크"""
        try:
            # 일일 손실 한도 체크
            if self.daily_pnl < -self.current_balance * self.daily_loss_limit:
                return False
                
            # 최대 드로다운 체크
            if self.max_drawdown > 0.2:  # 20% 최대 드로다운
                return False
                
            # 연속 손실 체크
            if self.consecutive_losses >= self.max_consecutive_losses:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"리스크 한도 체크 중 오류 발생: {e}")
            return False
            
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """변동성에 따른 포지션 크기 조정 계수"""
        try:
            if volatility > 0.03:
                return 0.5
            elif volatility > 0.02:
                return 0.7
            elif volatility < 0.01:
                return 1.2
            else:
                return 1.0
        except Exception as e:
            logger.error(f"변동성 계수 계산 중 오류 발생: {e}")
            return 1.0
            
    def _calculate_risk_factor(self, risk_level: str) -> float:
        """리스크 레벨에 따른 포지션 크기 조정 계수"""
        try:
            risk_factors = {
                'very_high': 0.5,
                'high': 0.7,
                'medium': 0.8,
                'low': 1.0
            }
            return risk_factors.get(risk_level, 0.5)
        except Exception as e:
            logger.error(f"리스크 계수 계산 중 오류 발생: {e}")
            return 0.5 