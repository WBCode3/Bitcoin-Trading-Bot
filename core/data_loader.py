"""
데이터 로딩을 위한 모듈
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.data = None
        logger.info("DataLoader 초기화됨")

    def load_market_data(self):
        """
        시장 데이터를 로드합니다.
        
        Returns:
            pd.DataFrame: 로드된 시장 데이터
        """
        try:
            # CSV 파일에서 데이터 로드
            file_path = 'data/btcusdt_1m_2022_to_2025.csv'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
            
            df = pd.read_csv(file_path)
            
            # 날짜 컬럼을 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 1분 데이터를 5분으로 리샘플링
            df_resampled = pd.DataFrame()
            df_resampled['open'] = df['open'].resample('5min').first()
            df_resampled['high'] = df['high'].resample('5min').max()
            df_resampled['low'] = df['low'].resample('5min').min()
            df_resampled['close'] = df['close'].resample('5min').last()
            df_resampled['volume'] = df['volume'].resample('5min').sum()
            
            # NaN 값 처리
            df_resampled = df_resampled.ffill().bfill()
            
            # 인덱스를 컬럼으로 변환
            df_resampled.reset_index(inplace=True)
            
            logger.info(f"시장 데이터 로드 완료: {len(df_resampled)}개의 데이터 포인트")
            return df_resampled
            
        except Exception as e:
            logger.error(f"시장 데이터 로드 중 오류 발생: {e}")
            raise

    def load_data(self, symbol, start_date=None, end_date=None):
        """
        주어진 기간 동안의 시장 데이터를 로드합니다.
        
        Args:
            symbol (str): 거래 심볼
            start_date (datetime, optional): 시작 날짜
            end_date (datetime, optional): 종료 날짜
            
        Returns:
            pd.DataFrame: 로드된 시장 데이터
        """
        try:
            # 여기에 실제 데이터 로딩 로직을 구현하세요
            # 예시로 더미 데이터를 생성합니다
            dates = pd.date_range(start=start_date or '2023-01-01',
                                end=end_date or datetime.now(),
                                freq='D')
            
            self.data = pd.DataFrame({
                'date': dates,
                'open': [100] * len(dates),
                'high': [105] * len(dates),
                'low': [95] * len(dates),
                'close': [102] * len(dates),
                'volume': [1000000] * len(dates)
            })
            
            logger.info(f"{symbol}에 대한 데이터 로드 완료")
            return self.data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise 