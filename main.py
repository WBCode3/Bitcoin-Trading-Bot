import logging
from datetime import datetime, timedelta
from simulation.backtest import BacktestSimulator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log')
    ]
)
logger = logging.getLogger('main')

def main():
    try:
        logger.info("시뮬레이션 시작")
        
        # 시뮬레이션 기간 설정 (전체 데이터 사용)
        start_date = None
        end_date = None
        
        # 백테스트 시뮬레이터 초기화 (초기 자본금 500,000원)
        simulator = BacktestSimulator(initial_capital=500000.0)
        
        # 시뮬레이션 실행
        results = simulator.run_simulation(
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("시뮬레이션 완료")
        
    except Exception as e:
        logger.error(f"시뮬레이션 실행 중 오류 발생: {e}")
        raise  # 오류를 다시 발생시켜 스택 트레이스를 확인할 수 있도록 함

if __name__ == "__main__":
    main() 