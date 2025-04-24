from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import settings
from utils.logger import setup_logger
from .exchange import ExchangeInterface

logger = setup_logger(__name__)

class RiskManager:
    def __init__(self, exchange: ExchangeInterface):
        self.exchange = exchange
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.trade_history = []
        self.volatility_history = []

    def check_daily_loss_limit(self) -> bool:
        """일일 손실 한도 체크"""
        if (datetime.now() - self.last_reset).days >= 1:
            self.daily_pnl = 0.0
            self.last_reset = datetime.now()
        
        if self.daily_pnl <= -settings.MAX_DAILY_LOSS:
            logger.warning(f"일일 손실 한도 도달: {self.daily_pnl:.2%}")
            return False
        return True

    def calculate_position_size(self, price: float) -> float:
        """변동성 기반 포지션 사이즈 계산"""
        try:
            # OHLCV 데이터 가져오기
            ohlcv = self.exchange.get_ohlcv('1h', 24)  # 24시간 데이터
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 변동성 계산
            vol_range = (df['high'].max() - df['low'].min()) / df['close'].iloc[-1]
            vol_std = df['close'].pct_change().rolling(14).std().iloc[-1]
            vol = max(vol_range, vol_std)
            
            # 변동성 기록
            self.volatility_history.append(vol)
            if len(self.volatility_history) > 100:
                self.volatility_history.pop(0)
            
            # 변동성 평균
            avg_vol = np.mean(self.volatility_history)
            
            # 잔고 조회
            balance = self.exchange.get_balance()['free']
            
            # 변동성에 따른 포지션 크기 조정
            if vol < avg_vol * 0.5:  # 변동성이 평균의 50% 미만
                pct = 0.8
            elif vol < avg_vol:  # 변동성이 평균 미만
                pct = 0.6
            elif vol < avg_vol * 1.5:  # 변동성이 평균의 150% 미만
                pct = 0.4
            else:  # 변동성이 평균의 150% 이상
                pct = 0.2
                
            # 최대 포지션 크기 제한
            max_position = balance * settings.MAX_POSITION_SIZE
            position_size = (balance * pct) / price
            
            # 레버리지 조정
            leverage = self.calculate_leverage(vol)
            position_size *= leverage
            
            return min(position_size, max_position)
            
        except Exception as e:
            logger.error(f"포지션 사이즈 계산 실패: {e}")
            return 0.0

    def calculate_leverage(self, volatility: float) -> float:
        """변동성 기반 레버리지 계산"""
        try:
            # 기본 레버리지
            base_leverage = settings.LEVERAGE
            
            # 변동성에 따른 레버리지 조정
            if volatility < 0.005:  # 매우 낮은 변동성
                return min(base_leverage * 2, 20)  # 최대 20배
            elif volatility < 0.01:  # 낮은 변동성
                return min(base_leverage * 1.5, 15)  # 최대 15배
            elif volatility < 0.02:  # 중간 변동성
                return base_leverage
            elif volatility < 0.03:  # 높은 변동성
                return max(base_leverage * 0.5, 5)  # 최소 5배
            else:  # 매우 높은 변동성
                return max(base_leverage * 0.25, 3)  # 최소 3배
                
        except Exception as e:
            logger.error(f"레버리지 계산 실패: {e}")
            return settings.LEVERAGE

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

    def get_risk_metrics(self) -> Dict[str, float]:
        """리스크 지표 반환"""
        return {
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'peak_balance': self.peak_balance,
            'total_trades': len(self.trade_history),
            'win_rate': self.calculate_win_rate(),
            'avg_volatility': np.mean(self.volatility_history) if self.volatility_history else 0
        }

    def calculate_win_rate(self) -> float:
        """승률 계산"""
        if not self.trade_history:
            return 0.0
            
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        return winning_trades / len(self.trade_history) 