import ccxt
from typing import Dict, Any, Optional, Tuple, List
from config.settings import settings
from utils.logger import setup_logger
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class Exchange:
    def __init__(self, api_key: str = None, api_secret: str = None):
        # 로거 설정
        self.logger = setup_logger('exchange')
        
        # API 키 설정
        self.api_key = api_key or settings['api']['binance']['api_key']
        self.api_secret = api_secret or settings['api']['binance']['api_secret']
        
        # 클라이언트 초기화
        self.client = Client(self.api_key, self.api_secret)
        self.symbol = 'BTCUSDT'  # 선물 거래 심볼 형식
        self.interval = '1m'
        self.leverage = settings['trading']['leverage']  # settings.py의 레버리지 값 사용
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
                self.logger.info(f"마진 타입 설정 완료: {self.margin_type}")
            except Exception as e:
                if "No need to change margin type" in str(e):
                    self.logger.info(f"이미 {self.margin_type} 마진 타입이 설정되어 있습니다.")
                else:
                    raise e
                    
            # 레버리지 설정
            try:
                self.client.futures_change_leverage(
                    symbol=self.symbol,
                    leverage=self.leverage
                )
                self.logger.info(f"레버리지 설정 완료: {self.leverage}x")
            except Exception as e:
                self.logger.error(f"레버리지 설정 중 오류 발생: {e}")
                raise e
                
        except Exception as e:
            self.logger.error(f"거래소 설정 중 오류 발생: {e}")
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
            self.logger.error(f"잔고 조회 중 오류 발생: {e}")
            return {'total': 0.0, 'free': 0.0}

    def get_positions(self) -> List[Dict[str, Any]]:
        """모든 포지션 정보 조회"""
        try:
            positions = self.client.futures_position_information()
            return [pos for pos in positions if float(pos['positionAmt']) != 0]
        except BinanceAPIException as e:
            self.logger.error(f"포지션 정보 조회 중 오류 발생: {e}")
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
            self.logger.error(f"포지션 정보 조회 중 오류 발생: {e}")
            return None

    def create_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """주문 생성"""
        try:
            # 주문 수량 포맷팅
            formatted_quantity = self.format_quantity(quantity)
            if formatted_quantity <= 0:
                self.logger.error(f"주문 수량이 0 이하입니다: {formatted_quantity}")
                return None

            # 주문 파라미터 설정
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': formatted_quantity
            }

            # 지정가 주문인 경우 가격 추가
            if order_type == 'LIMIT':
                if price is None:
                    self.logger.error("지정가 주문에는 가격이 필요합니다.")
                    return None
                params['price'] = self.format_price(price)
                params['timeInForce'] = 'GTC'

            # 주문 생성
            order = self.client.futures_create_order(**params)
            
            # 주문 정보 로깅
            self.logger.info(f"주문 생성 성공:")
            self.logger.info(f"- 심볼: {symbol}")
            self.logger.info(f"- 방향: {side}")
            self.logger.info(f"- 종류: {order_type}")
            self.logger.info(f"- 수량: {formatted_quantity}")
            if price is not None:
                self.logger.info(f"- 가격: {price}")
            self.logger.info(f"- 주문 ID: {order['orderId']}")
            
            return order

        except BinanceAPIException as e:
            self.logger.error(f"주문 생성 중 API 오류 발생: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"주문 생성 중 오류 발생: {str(e)}")
            return None

    def format_quantity(self, quantity: float) -> float:
        """주문 수량 포맷팅"""
        try:
            # 심볼 정보 가져오기
            symbol_info = self.client.futures_exchange_info()
            symbol_filter = next(filter(lambda x: x['symbol'] == self.symbol, symbol_info['symbols']))
            lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_filter['filters']))
            
            # 수량 정밀도 계산
            step_size = float(lot_size_filter['stepSize'])
            precision = int(round(-math.log(step_size, 10)))
            
            # 수량 포맷팅 (소수점 자릿수 제한)
            formatted_quantity = float(round(quantity, precision))
            
            # 최소 주문 수량 체크
            min_qty = float(lot_size_filter['minQty'])
            if formatted_quantity < min_qty:
                self.logger.error(f"주문 수량이 최소 주문 수량보다 작습니다: {formatted_quantity} < {min_qty}")
                return 0.0
            
            # 최대 주문 수량 체크
            max_qty = float(lot_size_filter['maxQty'])
            if formatted_quantity > max_qty:
                self.logger.error(f"주문 수량이 최대 주문 수량보다 큽니다: {formatted_quantity} > {max_qty}")
                return 0.0
            
            # 로깅
            self.logger.info(f"주문 수량 포맷팅:")
            self.logger.info(f"- 원본 수량: {quantity}")
            self.logger.info(f"- 정밀도: {precision}")
            self.logger.info(f"- 최소 수량: {min_qty}")
            self.logger.info(f"- 최대 수량: {max_qty}")
            self.logger.info(f"- 포맷팅된 수량: {formatted_quantity}")
            
            return formatted_quantity
            
        except Exception as e:
            self.logger.error(f"주문 수량 포맷팅 중 오류 발생: {str(e)}")
            return 0.0

    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            self.client.futures_cancel_order(orderId=order_id, symbol=self.symbol)
            self.logger.info(f"주문 취소 성공: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {e}")
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
            self.logger.error(f"OHLCV 데이터 조회 중 오류 발생: {e}")
            return []
        except Exception as e:
            self.logger.error(f"예상치 못한 오류 발생: {e}")
            return []

    def calculate_liquidation_price(self, entry_price: float, leverage: float, side: str) -> float:
        """청산가 계산"""
        try:
            if side == 'LONG':
                return entry_price * (1 - 1/leverage)
            else:
                return entry_price * (1 + 1/leverage)
        except Exception as e:
            self.logger.error(f"청산가 계산 중 오류 발생: {e}")
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
                    self.logger.info("이미 ISOLATED 마진 타입이 설정되어 있습니다.")
                else:
                    raise e
                    
            # 레버리지 설정
            self.set_leverage(1)  # 기본 레버리지 1x
            
            return True
            
        except Exception as e:
            self.logger.error(f"거래소 설정 중 오류 발생: {e}")
            return False

    def set_leverage(self, leverage: int) -> None:
        """레버리지 설정"""
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            self.logger.info(f"레버리지 설정 완료: {leverage}x")
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

    async def get_market_data(self) -> Dict[str, Any]:
        """현재 시장 데이터 조회"""
        try:
            # OHLCV 데이터 조회
            klines = await self.get_ohlcv(limit=100)
            if not klines:
                raise Exception("OHLCV 데이터 조회 실패")

            # OHLCV 데이터를 DataFrame으로 변환
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 현재가 조회
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])

            # 24시간 변동성
            ticker_24h = self.client.futures_ticker(symbol=self.symbol)
            price_change_24h = float(ticker_24h['priceChangePercent'])

            # 거래량
            volume_24h = float(ticker_24h['volume'])

            # 포지션 정보
            position = self.get_position()

            # 잔고 정보
            balance = self.get_balance()

            # 데이터 정리
            market_data = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'current_price': current_price,
                'df': df,  # DataFrame으로 변환된 OHLCV 데이터
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                'position': position,
                'balance': balance
            }

            return market_data

        except Exception as e:
            logger.error(f"시장 데이터 조회 중 오류 발생: {e}")
            raise

    def get_available_symbols(self, min_balance: float = 5.0) -> List[Dict[str, Any]]:
        """거래 가능한 선물 거래 쌍 목록 조회"""
        try:
            # 거래소 정보 조회
            exchange_info = self.client.futures_exchange_info()
            
            # 거래 가능한 심볼 필터링
            available_symbols = []
            for symbol in exchange_info['symbols']:
                if symbol['status'] == 'TRADING' and symbol['contractType'] == 'PERPETUAL':
                    # 최소 주문 수량과 가격 계산
                    min_qty = float(symbol['filters'][1]['minQty'])  # LOT_SIZE 필터
                    min_price = float(symbol['filters'][0]['minPrice'])  # PRICE_FILTER
                    
                    # 최소 주문 금액 계산
                    min_order_value = min_qty * min_price
                    
                    # 레버리지 고려한 최소 마진 계산
                    min_margin = min_order_value / self.leverage
                    
                    # 최소 마진이 사용자 잔고보다 작은 경우만 포함
                    if min_margin <= min_balance:
                        available_symbols.append({
                            'symbol': symbol['symbol'],
                            'base_asset': symbol['baseAsset'],
                            'quote_asset': symbol['quoteAsset'],
                            'price_precision': symbol['pricePrecision'],
                            'quantity_precision': symbol['quantityPrecision'],
                            'min_qty': min_qty,
                            'min_price': min_price,
                            'tick_size': float(symbol['filters'][0]['tickSize']),  # PRICE_FILTER
                            'min_margin': min_margin
                        })
            
            # 상세 정보 로깅
            self.logger.info(f"거래 가능한 코인 수: {len(available_symbols)}")
            self.logger.info(f"최소 자본금: {min_balance} USDT")
            self.logger.info(f"레버리지: {self.leverage}x")
            
            # 최소 마진 기준으로 정렬
            available_symbols.sort(key=lambda x: x['min_margin'])
            
            # 상위 10개 코인 로깅
            for symbol in available_symbols[:10]:
                self.logger.info(f"- {symbol['symbol']}: 최소 마진 {symbol['min_margin']:.4f} USDT")
            
            return available_symbols
            
        except Exception as e:
            self.logger.error(f"거래 가능한 코인 목록 조회 중 오류 발생: {str(e)}")
            return [] 