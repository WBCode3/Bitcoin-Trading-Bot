import ccxt
import pandas as pd
import time
from ta.momentum import RSIIndicator
from datetime import datetime

# 바이낸스 선물 연결
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

symbol = 'BTC/USDT'

def fetch_rsi_loop():
    while True:
        try:
            # 1분봉 데이터 수집
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamp')
            df = df.dropna(subset=['close'])

            # RSI 계산
            rsi = RSIIndicator(df['close'], window=14).rsi()

            # 출력
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n⏰ {now} - 1분봉 데이터")
            print("📉 최근 5개 Close:", df['close'].tail(5).tolist())
            print("📊 최근 5개 RSI:", rsi.tail(5).tolist())
            print(f"⭐ 현재 RSI: {rsi.iloc[-1]:.2f}")

        except Exception as e:
            print(f"에러 발생: {e}")

        time.sleep(60)  # 60초 대기

# 실행
fetch_rsi_loop()
