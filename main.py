import asyncio
import os
from datetime import datetime
from core.exchange import Exchange
from core.strategy import TradingStrategy, StrategyConfig
from core.risk_manager import RiskManager
from core.market_analyzer import MarketAnalyzer, MarketConfig
from utils.notifier import Notifier
from config.settings import settings
from utils.logger import setup_logger
import time
import sys

# 메인 로거 설정
logger = setup_logger('main')

class LiveTrader:
    def __init__(self):
        self.exchange = Exchange(
            api_key=settings['api']['binance']['api_key'],
            api_secret=settings['api']['binance']['api_secret']
        )
        self.strategy = TradingStrategy(
            config=StrategyConfig(
                min_signal_interval=300,  # 5분
                max_consecutive_losses=3,
                leverage=settings['trading']['leverage']
            )
        )
        self.risk_manager = RiskManager()  # 기본 설정으로 초기화
        self.market_analyzer = MarketAnalyzer(
            config=MarketConfig(
                rsi_period=14,
                bb_period=20,
                bb_std=2,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                adx_period=14,
                vwap_period=20,
                volatility_threshold=0.02,
                volume_threshold=1.5
            )
        )
        self.notifier = Notifier()
        
        # 거래 설정
        balance = self.exchange.get_balance()
        self.initial_capital = balance['total'] if balance['total'] > 0 else settings['trading']['initial_capital']
        self.leverage = settings['trading']['leverage']
        self.position_size = settings['trading']['position_size']
        
        logger.info(f"실전 매매 시작 - 초기 자본금: {self.initial_capital}, 레버리지: {self.leverage}x")

    def live_trade(self):
        """실전 매매 루프"""
        try:
            while True:
                # 시장 데이터 수집
                market_data = self.exchange.get_market_data()
                df = market_data['df']
                current_price = market_data['current_price']
                
                # 지표 계산
                indicators = self.strategy.calculate_indicators(df)
                market_data['indicators'] = indicators
                
                # 진입 신호 확인
                signal = self.strategy.generate_signal(market_data)
                action = signal.get('action')
                confidence = signal.get('confidence', 0)
                reason = signal.get('reason', '')
                
                logger.info(f"[신호] {action.upper()} | 신뢰도: {confidence:.2f} | 사유: {reason}")
                
                # 진입 조건: BUY 또는 SELL 신호 + 신뢰도 0.7 이상
                if action in ['buy', 'sell'] and confidence >= 0.7:
                    # 포지션 크기 계산
                    atr = indicators.get('atr', df['high'][-14:].max() - df['low'][-14:].min())
                    position_size = self.strategy.calculate_position_size(current_price, atr)
                    if position_size <= 0:
                        logger.warning("유효하지 않은 포지션 크기. 매매 스킵.")
                        time.sleep(10)
                        continue
                    # 주문 실행
                    order = self.exchange.create_order(
                        symbol=self.exchange.symbol,
                        side='BUY' if action == 'buy' else 'SELL',
                        order_type='MARKET',
                        quantity=position_size
                    )
                    logger.info(f"주문 실행: {order}")
                else:
                    logger.info("진입 신호 없음. 10초 후 재시도.")
                    time.sleep(10)
        except Exception as e:
            logger.error(f"실전 매매 중 오류 발생: {e}")

    def start(self):
        self.notifier.send_message(
            f"🚀 <b>트레이딩 봇 시작</b>\n"
            f"초기 자본금: {self.initial_capital:,.0f}원\n"
            f"레버리지: {self.leverage}x\n"
            f"포지션 크기: {self.position_size*100:.0f}%"
        )
        # 실전 매매 실행
        self.live_trade()

async def test_trade():
    """실제 전략 기반 거래 테스트"""
    try:
        # 거래소 초기화
        exchange = Exchange()
        await exchange.initialize()
        
        # 레버리지 설정
        await exchange.set_leverage(30)
        
        # 초기 계좌 잔고 확인
        initial_balance = await exchange.get_balance()
        logger.info(f"초기 계좌 잔고: {initial_balance:.2f} USDT")
        
        # 현재 시장 데이터 수집
        market_data = await exchange.get_market_data()
        current_price = market_data['current_price']
        
        # 포지션 크기 계산
        atr = market_data['indicators']['atr'].iloc[-1]
        position_size = exchange.strategy.calculate_position_size(current_price, atr)
        position_value = position_size * current_price
        position_percentage = (position_value / initial_balance) * 100
        
        # 알림봇으로 시작 메시지 전송
        message = f"""🚀 <b>트레이딩 봇 시작</b>
초기 자본금: {initial_balance:.2f} USDT
레버리지: 30x
포지션 크기: {position_percentage:.1f}% (ATR 기반 동적 계산)"""
        await send_telegram_message(message)
        
        # 매수 주문 실행
        order = await exchange.create_order(
            symbol=exchange.symbol,
            side='BUY',
            quantity=position_size,
            order_type='MARKET'
        )
        
        if not order:
            logger.error("매수 주문 실패")
            return
            
        logger.info(f"매수 주문 실행: {position_size} BTC @ {current_price}")
        
        # 1분 대기
        await asyncio.sleep(60)
        
        # 매도 주문 실행
        exit_order = await exchange.create_order(
            symbol=exchange.symbol,
            side='SELL',
            quantity=position_size,
            order_type='MARKET'
        )
        
        if not exit_order:
            logger.error("매도 주문 실패")
            return
            
        # 최종 계좌 잔고 확인
        final_balance = await exchange.get_balance()
        logger.info(f"최종 계좌 잔고: {final_balance:.2f} USDT")
        
    except Exception as e:
        logger.error(f"테스트 거래 중 오류 발생: {e}")
        await send_telegram_message(f"❌ 오류 발생: {str(e)}")

def main():
    trader = LiveTrader()
    trader.start()

if __name__ == "__main__":
    main() 