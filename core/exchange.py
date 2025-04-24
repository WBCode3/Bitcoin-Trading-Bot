import ccxt
from typing import Dict, Any, Optional, Tuple, List
from config.settings import settings
from utils.logger import setup_logger
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Exchange:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.client = Client(api_key, api_secret) if api_key and api_secret else None
        self.symbol = 'BTCUSDT'  # 선물 거래 심볼 형식
        self.interval = '1m'
        self.leverage = 1  # 기본 레버리지 1x로 변경
        self.margin_type = 'ISOLATED'
        
        # 거래소 설정
        if self.client:
            self._setup_exchange()
        
    def _setup_exchange(self):
        """거래소 초기 설정"""
        try:
            # 마진 타입 설정
            try:
                self.client.futures_change_margin_type(
                    symbol=self.symbol,
                    marginType=self.margin_type
                )
                logger.info(f"마진 타입 설정 완료: {self.margin_type}")
            except Exception as e:
                if "No need to change margin type" in str(e):
                    logger.info(f"이미 {self.margin_type} 마진 타입이 설정되어 있습니다.")
                else:
                    raise e
                    
            # 레버리지 설정
            try:
                self.client.futures_change_leverage(
                    symbol=self.symbol,
                    leverage=self.leverage
                )
                logger.info(f"레버리지 설정 완료: {self.leverage}x")
            except Exception as e:
                logger.error(f"레버리지 설정 중 오류 발생: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"거래소 설정 중 오류 발생: {e}")
            raise e

    def get_balance(self) -> Dict[str, float]:
        """현재 잔고 조회"""
        try:
            account = self.client.futures_account()
            total_balance = float(account['totalWalletBalance'])
            free_balance = float(account['availableBalance'])
            
            return {
                'total': total_balance,
                'free': free_balance
            }
        except BinanceAPIException as e:
            logger.error(f"잔고 조회 중 오류 발생: {e}")
            return {'total': 0.0, 'free': 0.0}

    def get_positions(self) -> List[Dict[str, Any]]:
        """모든 포지션 정보 조회"""
        try:
            positions = self.client.futures_position_information()
            return [pos for pos in positions if float(pos['positionAmt']) != 0]
        except BinanceAPIException as e:
            logger.error(f"포지션 정보 조회 중 오류 발생: {e}")
            return []

    def get_position(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """특정 심볼의 포지션 정보 조회"""
        try:
            symbol = symbol or self.symbol
            positions = self.client.futures_position_information(symbol=symbol)
            if positions and float(positions[0]['positionAmt']) != 0:
                return positions[0]
            return None
        except BinanceAPIException as e:
            logger.error(f"포지션 정보 조회 중 오류 발생: {e}")
            return None

    def create_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET') -> Dict[str, Any]:
        """주문 생성"""
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            logger.info(f"주문 생성 성공: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"주문 생성 중 오류 발생: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            self.client.futures_cancel_order(orderId=order_id, symbol=self.symbol)
            logger.info(f"주문 취소 성공: {order_id}")
            return True
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return False

    async def get_ohlcv(self, symbol: str = None, interval: str = None, limit: int = 100) -> List[List[Any]]:
        """OHLCV 데이터 조회"""
        try:
            symbol = symbol or self.symbol
            interval = interval or self.interval
            
            # 인터벌 검증
            valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            if interval not in valid_intervals:
                interval = '1m'
            
            # 심볼 검증
            if not isinstance(symbol, str) or not symbol:
                symbol = 'BTCUSDT'
            elif '/' in symbol:
                symbol = symbol.replace('/', '')
            
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if not klines:
                return []
            
            # 데이터 형식 변환
            formatted_klines = []
            for k in klines:
                try:
                    formatted_klines.append([
                        int(k[0]),      # timestamp
                        float(k[1]),    # open
                        float(k[2]),    # high
                        float(k[3]),    # low
                        float(k[4]),    # close
                        float(k[5])     # volume
                    ])
                except (ValueError, IndexError):
                    continue
            
            if not formatted_klines:
                return []
                
            return formatted_klines
            
        except BinanceAPIException as e:
            logger.error(f"OHLCV 데이터 조회 중 오류 발생: {e}")
            return []
        except Exception as e:
            logger.error(f"예상치 못한 오류 발생: {e}")
            return []

    def calculate_liquidation_price(self, entry_price: float, leverage: float, side: str) -> float:
        """청산가 계산"""
        try:
            if side == 'LONG':
                return entry_price * (1 - 1/leverage)
            else:
                return entry_price * (1 + 1/leverage)
        except Exception as e:
            logger.error(f"청산가 계산 중 오류 발생: {e}")
            return 0.0

    def initialize(self) -> bool:
        """거래소 초기화"""
        try:
            # 마진 타입 설정 (isolated)
            try:
                self.client.futures_change_margin_type(
                    symbol='BTCUSDT',
                    marginType='ISOLATED'
                )
            except Exception as e:
                if "No need to change margin type" in str(e):
                    logger.info("이미 ISOLATED 마진 타입이 설정되어 있습니다.")
                else:
                    raise e
                    
            # 레버리지 설정
            self.set_leverage(1)  # 기본 레버리지 1x
            
            return True
            
        except Exception as e:
            logger.error(f"거래소 설정 중 오류 발생: {e}")
            return False

    def set_leverage(self, leverage: int) -> None:
        """레버리지 설정"""
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            logger.info(f"레버리지 설정 완료: {leverage}x")
        except Exception as e:
            logger.error(f"레버리지 설정 중 오류 발생: {e}")
            raise

    async def get_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """히스토리컬 데이터 조회 (시뮬레이션용 테스트 데이터 생성)"""
        try:
            # 시작일과 종료일 파싱
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # 날짜 범위 생성 (1분 간격)
            dates = pd.date_range(start=start, end=end, freq='1min')
            
            # 기본 가격 설정
            base_price = 90000.0
            
            # 가격 변동 시뮬레이션
            np.random.seed(42)  # 재현성을 위한 시드 설정
            returns = np.random.normal(0.0001, 0.002, size=len(dates))  # 평균 수익률과 변동성 조정
            price_multipliers = np.exp(np.cumsum(returns))
            
            # 가격 데이터 생성
            prices = base_price * price_multipliers
            high_prices = prices * (1 + np.random.uniform(0, 0.003, size=len(dates)))
            low_prices = prices * (1 - np.random.uniform(0, 0.003, size=len(dates)))
            
            # 거래량 생성
            volumes = np.random.lognormal(10, 1, size=len(dates))
            
            # DataFrame 생성
            df = pd.DataFrame({
                'open': prices,
                'high': high_prices,
                'low': low_prices,
                'close': prices,
                'volume': volumes
            }, index=dates)
            
            return df
            
        except Exception as e:
            logger.error(f"히스토리컬 데이터 생성 중 오류 발생: {e}")
            return pd.DataFrame() 