from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def calculate_volatility(df: pd.DataFrame, window: int = 10) -> Tuple[float, float]:
    """변동성 계산"""
    try:
        # Range 기반 변동성
        vol_range = (df['high'].max() - df['low'].min()) / df['close'].iloc[-1]
        
        # StdDev 기반 변동성
        vol_std = df['close'].pct_change().rolling(window).std().iloc[-1]
        
        return vol_range, vol_std
        
    except Exception as e:
        logger.error(f"변동성 계산 실패: {e}")
        return 0.0, 0.0

def calculate_position_size(balance: float, price: float, volatility: float, 
                          avg_volatility: float) -> float:
    """변동성 기반 포지션 사이즈 계산"""
    try:
        # 변동성에 따른 포지션 크기 조정
        if volatility < avg_volatility * 0.5:  # 변동성이 평균의 50% 미만
            pct = 0.8
        elif volatility < avg_volatility:  # 변동성이 평균 미만
            pct = 0.6
        elif volatility < avg_volatility * 1.5:  # 변동성이 평균의 150% 미만
            pct = 0.4
        else:  # 변동성이 평균의 150% 이상
            pct = 0.2
            
        # 최대 포지션 크기 제한
        max_position = balance * settings.MAX_POSITION_SIZE
        position_size = (balance * pct) / price
        
        return min(position_size, max_position)
        
    except Exception as e:
        logger.error(f"포지션 사이즈 계산 실패: {e}")
        return 0.0

def calculate_leverage(volatility: float) -> float:
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

def format_trade_info(trade: Dict[str, Any]) -> str:
    """거래 정보 포맷팅"""
    try:
        if trade['type'] == 'entry':
            return (f"진입: {trade['side']} {trade['size']:.4f} @ {trade['price']:.2f} "
                   f"(수수료: {trade['fee']:.2f} USDT)")
        else:
            return (f"청산: {trade['side']} {trade['size']:.4f} @ {trade['price']:.2f} "
                   f"(PnL: {trade['pnl']:.2%}, 수수료: {trade['fee']:.2f} USDT)")
                   
    except Exception as e:
        logger.error(f"거래 정보 포맷팅 실패: {e}")
        return str(trade)

def format_metrics(metrics: Dict[str, Any]) -> str:
    """성과 지표 포맷팅"""
    try:
        return (f"총 수익률: {metrics['total_return']:.2%}\n"
                f"샤프비율: {metrics['sharpe_ratio']:.2f}\n"
                f"승률: {metrics['win_rate']:.2%}\n"
                f"평균 수익: {metrics['avg_profit']:.2%}\n"
                f"평균 손실: {metrics['avg_loss']:.2%}\n"
                f"최대 드로다운: {metrics['max_drawdown']:.2%}\n"
                f"총 거래 횟수: {metrics['total_trades']}\n"
                f"수익 요인: {metrics['profit_factors']}")
                
    except Exception as e:
        logger.error(f"성과 지표 포맷팅 실패: {e}")
        return str(metrics)

def check_market_condition(df: pd.DataFrame) -> str:
    """시장 상황 판단"""
    try:
        # RSI
        rsi = RSIIndicator(df['close']).rsi().iloc[-1]
        
        # MACD
        macd = MACD(df['close']).macd_diff().iloc[-1]
        
        # 볼린저 밴드
        bb = BollingerBands(df['close'])
        bb_width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / bb.bollinger_mavg().iloc[-1]
        
        # 변동성
        vol_range, vol_std = calculate_volatility(df)
        volatility = max(vol_range, vol_std)
        
        # 시장 상황 판단
        if volatility > 0.03:  # 매우 높은 변동성
            return 'high_volatility'
        elif bb_width > 0.02:  # 넓은 밴드
            return 'wide_range'
        elif rsi < 30:  # 과매도
            return 'oversold'
        elif rsi > 70:  # 과매수
            return 'overbought'
        else:
            return 'normal'
            
    except Exception as e:
        logger.error(f"시장 상황 판단 실패: {e}")
        return 'unknown' 