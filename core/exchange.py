import ccxt
from typing import Dict, Any, Optional, Tuple
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ExchangeInterface:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': settings.API_KEY,
            'secret': settings.API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self._setup_exchange()

    def _setup_exchange(self):
        """거래소 초기 설정"""
        try:
            # 레버리지 설정
            self.exchange.set_leverage(settings.LEVERAGE, settings.SYMBOL)
            # 마진 모드 설정
            self.exchange.set_margin_mode(settings.MARGIN_MODE, settings.SYMBOL)
            logger.info(f"거래소 설정 완료: 레버리지={settings.LEVERAGE}x, 마진모드={settings.MARGIN_MODE}")
        except Exception as e:
            logger.error(f"거래소 설정 실패: {e}")

    def get_balance(self) -> Dict[str, float]:
        """계좌 잔고 조회"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total': balance['total']['USDT'],
                'free': balance['free']['USDT'],
                'used': balance['used']['USDT']
            }
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return {'total': 0, 'free': 0, 'used': 0}

    def get_position(self) -> Tuple[Optional[str], float, float, float]:
        """현재 포지션 정보 조회"""
        try:
            positions = self.exchange.fetch_positions([settings.SYMBOL])
            if positions:
                position = positions[0]
                side = 'long' if float(position['contracts']) > 0 else 'short' if float(position['contracts']) < 0 else None
                return (
                    side,
                    float(position['entryPrice']),
                    float(position['leverage']),
                    abs(float(position['contracts']))
                )
            return None, 0, settings.LEVERAGE, 0
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return None, 0, settings.LEVERAGE, 0

    def create_order(self, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """주문 생성"""
        try:
            if price:
                order = self.exchange.create_limit_order(
                    settings.SYMBOL,
                    side,
                    amount,
                    price
                )
            else:
                order = self.exchange.create_market_order(
                    settings.SYMBOL,
                    side,
                    amount
                )
            logger.info(f"주문 생성 성공: {order}")
            return order
        except Exception as e:
            logger.error(f"주문 생성 실패: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            self.exchange.cancel_order(order_id, settings.SYMBOL)
            logger.info(f"주문 취소 성공: {order_id}")
            return True
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return False

    def get_ohlcv(self, timeframe: str = '1m', limit: int = 100) -> list:
        """OHLCV 데이터 조회"""
        try:
            return self.exchange.fetch_ohlcv(settings.SYMBOL, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"OHLCV 데이터 조회 실패: {e}")
            return []

    def calculate_liquidation_price(self, side: str, entry_price: float, leverage: float) -> float:
        """청산가 계산"""
        if side == 'long':
            return entry_price * (1 - 1/leverage)
        else:
            return entry_price * (1 + 1/leverage) 