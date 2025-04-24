import asyncio
import pandas as pd
from datetime import datetime, timedelta
from config.settings import settings
from utils.logger import setup_logger
from utils.notifier import Notifier
from core.exchange import ExchangeInterface
from core.risk_manager import RiskManager
from core.strategy import TradingStrategy
from core.backtest import BacktestEngine
from utils import format_metrics, format_trade_info

logger = setup_logger(__name__)

class TradingBot:
    def __init__(self):
        self.exchange = ExchangeInterface()
        self.risk_manager = RiskManager(self.exchange)
        self.strategy = TradingStrategy(self.exchange, self.risk_manager)
        self.notifier = Notifier()
        self.running = False

    async def start(self):
        """봇 시작"""
        try:
            self.running = True
            logger.info("트레이딩 봇 시작")
            self.notifier.send_message("트레이딩 봇이 시작되었습니다.")
            
            # 백테스트 실행
            if settings.RUN_BACKTEST:
                logger.info("백테스트 시작")
                
                # 과거 데이터 로드
                data1 = self.exchange.get_historical_data('1m', settings.BACKTEST_START, settings.BACKTEST_END)
                data5 = self.exchange.get_historical_data('5m', settings.BACKTEST_START, settings.BACKTEST_END)
                
                # 백테스트 엔진 초기화
                backtest = BacktestEngine(self.strategy)
                
                # 백테스트 실행
                results = backtest.run(data1, data5)
                
                # 결과 출력
                logger.info("백테스트 결과:")
                logger.info(format_metrics(results['metrics']))
                
                # 거래 내역 저장
                trades_df = pd.DataFrame(results['trades'])
                trades_df.to_csv('backtest_trades.csv', index=False)
                
                # 자본 곡선 저장
                equity_df = pd.DataFrame({'equity': results['equity_curve']})
                equity_df.to_csv('backtest_equity.csv', index=False)
                
                logger.info("백테스트 완료")
            
            # 실시간 트레이딩 실행
            if settings.RUN_LIVE:
                logger.info("실시간 트레이딩 시작")
                
                while True:
                    try:
                        # 거래 실행
                        self.strategy.execute_trade()
                        
                        # 파라미터 최적화 (매주 월요일)
                        if datetime.now().weekday() == 0 and datetime.now().hour == 1:
                            self.strategy.optimize_parameters()
                        
                        # 1분 대기
                        await asyncio.sleep(60)
                        
                    except Exception as e:
                        logger.error(f"트레이딩 실행 중 오류 발생: {e}")
                        await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"봇 실행 중 오류 발생: {e}")
            self.notifier.send_message(f"봇 실행 중 오류 발생: {e}")
        finally:
            self.stop()

    def stop(self):
        """봇 중지"""
        self.running = False
        logger.info("트레이딩 봇 중지")
        self.notifier.send_message("트레이딩 봇이 중지되었습니다.")

    def run_trading_cycle(self):
        """트레이딩 사이클 실행"""
        try:
            # 일일 손실 한도 체크
            if not self.risk_manager.check_daily_loss_limit():
                logger.warning("일일 손실 한도 도달로 거래 중지")
                return

            # 거래 실행
            self.strategy.execute_trade()
            
        except Exception as e:
            logger.error(f"트레이딩 사이클 실행 중 오류 발생: {e}")
            self.notifier.send_message(f"트레이딩 사이클 실행 중 오류 발생: {e}")

    def send_status_update(self):
        """상태 업데이트 전송"""
        try:
            # 계좌 정보
            balance = self.exchange.get_balance()
            position = self.exchange.get_position()
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # 메시지 구성
            message = (
                f"=== 트레이딩 봇 상태 업데이트 ===\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"계좌 잔고: {balance['total']:.2f} USDT\n"
                f"사용 가능: {balance['free']:.2f} USDT\n"
                f"포지션: {position[0] if position[0] else '없음'}\n"
                f"일일 P&L: {risk_metrics['daily_pnl']:.2%}\n"
                f"최대 드로다운: {risk_metrics['max_drawdown']:.2%}"
            )
            
            # 알림 전송
            self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"상태 업데이트 전송 중 오류 발생: {e}")

async def main():
    try:
        # 거래소 인터페이스 초기화
        exchange = ExchangeInterface()
        
        # 리스크 매니저 초기화
        risk_manager = RiskManager(exchange)
        
        # 전략 초기화
        strategy = TradingStrategy(exchange, risk_manager)
        
        # 백테스트 실행
        if settings.RUN_BACKTEST:
            logger.info("백테스트 시작")
            
            # 과거 데이터 로드
            data1 = exchange.get_historical_data('1m', settings.BACKTEST_START, settings.BACKTEST_END)
            data5 = exchange.get_historical_data('5m', settings.BACKTEST_START, settings.BACKTEST_END)
            
            # 백테스트 엔진 초기화
            backtest = BacktestEngine(strategy)
            
            # 백테스트 실행
            results = backtest.run(data1, data5)
            
            # 결과 출력
            logger.info("백테스트 결과:")
            logger.info(format_metrics(results['metrics']))
            
            # 거래 내역 저장
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv('backtest_trades.csv', index=False)
            
            # 자본 곡선 저장
            equity_df = pd.DataFrame({'equity': results['equity_curve']})
            equity_df.to_csv('backtest_equity.csv', index=False)
            
            logger.info("백테스트 완료")
        
        # 실시간 트레이딩 실행
        if settings.RUN_LIVE:
            logger.info("실시간 트레이딩 시작")
            
            while True:
                try:
                    # 거래 실행
                    strategy.execute_trade()
                    
                    # 파라미터 최적화 (매주 월요일)
                    if datetime.now().weekday() == 0 and datetime.now().hour == 1:
                        strategy.optimize_parameters()
                    
                    # 1분 대기
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"트레이딩 실행 중 오류 발생: {e}")
                    await asyncio.sleep(60)
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.start()) 