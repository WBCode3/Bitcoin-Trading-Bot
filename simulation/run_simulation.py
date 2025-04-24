import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from backtest import BacktestSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from core.strategy import TradingStrategy
from core.risk_manager import RiskManager
from core.market_analyzer import MarketAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """데이터 로드"""
    try:
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

def plot_results(results: dict):
    """결과 시각화"""
    try:
        # 일일 자본금 변화
        daily_results = results['daily_results']
        dates = [r['date'] for r in daily_results]
        capital = [r['capital'] for r in daily_results]
        
        plt.figure(figsize=(15, 8))
        plt.plot(dates, capital)
        plt.title('일일 자본금 변화')
        plt.xlabel('날짜')
        plt.ylabel('자본금 (원)')
        plt.grid(True)
        plt.savefig('simulation_results.png')
        plt.close()
        
        # 거래 통계
        stats = {
            '총 수익률': f"{results['total_return']*100:.2f}%",
            '연간 수익률': f"{results['annual_return']*100:.2f}%",
            '최대 드로다운': f"{results['max_drawdown']*100:.2f}%",
            '총 거래 횟수': results['total_trades'],
            '승률': f"{results['win_rate']*100:.2f}%"
        }
        
        print("\n=== 시뮬레이션 결과 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"결과 시각화 중 오류 발생: {e}")

async def main():
    # 로거 설정
    setup_logger('simulation')
    logger = logging.getLogger(__name__)
    
    try:
        # CSV 파일에서 데이터 로드
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'btcusdt_1m_2022_to_2025.csv')
        if not os.path.exists(data_path):
            logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            return
            
        data = load_data(data_path)
        if data.empty:
            logger.error("데이터 로드 실패")
            return
            
        logger.info(f"데이터 로드 완료: {len(data)}개의 데이터 포인트")
        
        # 초기 자본금 설정 (500,000원)
        initial_capital = 500000
        
        # 전략, 리스크 매니저, 마켓 애널라이저 초기화
        strategy = TradingStrategy()
        risk_manager = RiskManager()
        market_analyzer = MarketAnalyzer()
        
        # 백테스트 시뮬레이터 생성
        simulator = BacktestSimulator(
            initial_capital=initial_capital,
            trading_strategy=strategy,
            risk_manager=risk_manager,
            market_analyzer=market_analyzer
        )
        
        # 시뮬레이션 실행
        print("시뮬레이션을 시작합니다...")
        await simulator.run_simulation()
        print("시뮬레이션이 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"시뮬레이션 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 