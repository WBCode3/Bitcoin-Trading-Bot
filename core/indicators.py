import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1])

def calculate_macd(prices: pd.Series, params: Dict[str, int]) -> Dict[str, float]:
    """MACD 계산"""
    exp1 = prices.ewm(span=params['fast'], adjust=False).mean()
    exp2 = prices.ewm(span=params['slow'], adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=params['signal'], adjust=False).mean()
    histogram = macd - signal
    
    return {
        'macd': float(macd.iloc[-1]),
        'signal': float(signal.iloc[-1]),
        'histogram': float(histogram.iloc[-1])
    }

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
    """볼린저 밴드 계산"""
    middle = data['close'].rolling(window=period).mean()
    std_dev = data['close'].rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR 계산"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return float(atr.iloc[-1])

def calculate_stochastic(df: pd.DataFrame, params: Dict[str, int]) -> Dict[str, float]:
    """스토캐스틱 계산"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    lowest_low = low.rolling(window=params['k_period']).min()
    highest_high = high.rolling(window=params['k_period']).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=params['d_period']).mean()
    
    return {
        'k': float(k.iloc[-1]),
        'd': float(d.iloc[-1])
    }

def calculate_adx(df: pd.DataFrame, params: Dict[str, int]) -> Dict[str, float]:
    """ADX 계산"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=params['period']).mean()
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_di = 100 * (plus_dm.rolling(window=params['period']).mean() / atr)
    minus_di = 100 * (abs(minus_dm.rolling(window=params['period']).mean()) / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=params['period']).mean()
    
    return {
        'adx': float(adx.iloc[-1]),
        'plus_di': float(plus_di.iloc[-1]),
        'minus_di': float(minus_di.iloc[-1])
    }

def calculate_ichimoku(df: pd.DataFrame, params: Dict[str, int]) -> Dict[str, float]:
    """일목균형표 계산"""
    high = df['high']
    low = df['low']
    
    conversion = (high.rolling(window=params['conversion']).max() + 
                 low.rolling(window=params['conversion']).min()) / 2
    base = (high.rolling(window=params['base']).max() + 
            low.rolling(window=params['base']).min()) / 2
    span_a = (conversion + base) / 2
    span_b = (high.rolling(window=params['span_b']).max() + 
              low.rolling(window=params['span_b']).min()) / 2
    
    return {
        'conversion': float(conversion.iloc[-1]),
        'base': float(base.iloc[-1]),
        'span_a': float(span_a.iloc[-1]),
        'span_b': float(span_b.iloc[-1])
    } 