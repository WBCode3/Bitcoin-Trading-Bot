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

    async def test_trade(self):
        """테스트 매매 실행"""
        try:
            # 1. 시장 데이터 수집
            logger.info("시장 데이터 수집 중...")
            market_data = await self.exchange.get_market_data()
            if not market_data:
                logger.error("시장 데이터 수집 실패")
                return
            
            current_price = market_data['current_price']
            logger.info(f"현재가: {current_price}")
            
            # 2. 포지션 크기 계산 (최소 주문 금액 100 USDT 이상)
            position_size = 0.002  # 약 187 USDT (현재가 93672.4 기준)
            logger.info(f"매수 수량: {position_size} BTC (약 {position_size * current_price:.2f} USDT)")
            
            # 3. 매수 주문 실행
            logger.info("매수 주문 실행 중...")
            buy_order = await self.exchange.create_order(
                symbol=settings['trading']['symbol'],
                side='BUY',
                order_type='market',
                quantity=position_size
            )
            logger.info(f"매수 주문 완료: {buy_order}")
            
            # 4. 5분 대기
            logger.info("5분 대기 중...")
            await asyncio.sleep(300)
            
            # 5. 포지션 확인
            position = self.exchange.get_position()
            logger.info(f"현재 포지션: {position}")
            
            # 6. 매도 주문 실행
            logger.info("매도 주문 실행 중...")
            sell_order = await self.exchange.create_order(
                symbol=settings['trading']['symbol'],
                side='SELL',
                order_type='market',
                quantity=position['amount']
            )
            logger.info(f"매도 주문 완료: {sell_order}")
            
            # 7. 최종 포지션 확인
            final_position = self.exchange.get_position()
            logger.info(f"최종 포지션: {final_position}")
            
        except Exception as e:
            logger.error(f"테스트 매매 중 오류 발생: {e}")

    async def start(self):
        # 시작 메시지 전송
        self.notifier.send_message(
            f"🚀 <b>트레이딩 봇 시작</b>\n"
            f"초기 자본금: {self.initial_capital:,.0f}원\n"
            f"레버리지: {self.leverage}x\n"
            f"포지션 크기: {self.position_size*100:.0f}%"
        )
        
        # 테스트 매매 실행
        await self.test_trade()

async def main():
    trader = LiveTrader()
    await trader.start()

if __name__ == "__main__":
    asyncio.run(main()) 