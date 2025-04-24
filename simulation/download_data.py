import os
import sys
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import logging

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

logger = setup_logger('data_downloader')

def download_historical_data(symbol: str = 'BTCUSDT', interval: str = '1m', 
                           start_year: int = 2022, end_year: int = None):
    """바이낸스에서 과거 데이터 다운로드 (1년 단위)"""
    try:
        # 데이터 디렉토리 생성
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 바이낸스 클라이언트 초기화
        client = Client()
        
        if end_year is None:
            end_year = datetime.now().year
            
        all_data = []
        total_points = 0
        
        print("\n=== 데이터 다운로드 시작 ===")
        print(f"기간: {start_year}년 ~ {end_year}년")
        print("=" * 50)
        
        for year in range(start_year, end_year + 1):
            # 연도별 시작/종료 시간 설정
            start_time = int(datetime(year, 1, 1).timestamp() * 1000)
            if year == datetime.now().year:
                end_time = int(datetime.now().timestamp() * 1000)
            else:
                end_time = int(datetime(year, 12, 31, 23, 59, 59).timestamp() * 1000)
            
            print(f"\n[{year}년 데이터 다운로드 중...]")
            print(f"시작일: {datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d')}")
            print(f"종료일: {datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d')}")
            
            # 데이터 다운로드
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time
            )
            
            # DataFrame 생성
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # 데이터 전처리
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            all_data.append(df)
            total_points += len(df)
            
            print(f"\n{year}년 데이터 다운로드 완료!")
            print(f"- 데이터 포인트: {len(df):,}개")
            print(f"- 가격 범위: {df['low'].min():,.2f} ~ {df['high'].max():,.2f} USDT")
            print(f"- 평균 일일 거래량: {df['volume'].mean():,.2f} BTC")
            print("-" * 50)
            
        # 모든 데이터 합치기
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values('timestamp')
        
        # CSV 파일로 저장
        output_file = os.path.join(data_dir, f"{symbol.lower()}_1m_{start_year}_to_{end_year}.csv")
        final_df.to_csv(output_file, index=False)
        
        print("\n=== 최종 결과 ===")
        print(f"총 데이터 포인트: {total_points:,}개")
        print(f"기간: {final_df['timestamp'].min()} ~ {final_df['timestamp'].max()}")
        print(f"저장된 파일: {output_file}")
        print("=" * 50)
        
        return final_df
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    download_historical_data(start_year=2022) 