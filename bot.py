import os
import time
import json
import requests
import ccxt  # CCXT 라이브러리로 바이낸스 선물 연결
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from ta.momentum import RSIIndicator  # RSI 계산
from ta.volatility import BollingerBands  # 볼린저 밴드
from ta.trend import MACD  # MACD
from telegram.ext import Application, CommandHandler

# Fallback for scipy.stats.norm
try:
    from scipy.stats import norm
except ImportError:
    class norm:
        @staticmethod
        def rvs(size=None):
            import numpy as _np
            return _np.random.standard_normal(size)


class TradingBot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret = os.getenv('BINANCE_SECRET_KEY')
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK')
        self.backup_file = 'bot_state.json'
        self.historical_csv = os.getenv('HISTORICAL_CSV', 'historical.csv')
        self.symbol = 'BTC/USDT'

        # 전략 기본 파라미터
        self.best_params = {
            'rsi_long': 25,
            'rsi_short': 75,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.10
        }
        self.max_profit_price = None
        self.trail_active = False

        # 수수료·슬리피지
        self.fee_rate = 0.0004  # 0.04%
        self.slippage_pct = 0.0005  # ±0.05%

        # 거래소 연결
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # 텔레그램 커맨드
        self.application = Application.builder().token(self.telegram_token).build()
        self.application.add_handler(CommandHandler('status', self.cmd_status))
        self.application.add_handler(CommandHandler('optimize', self.cmd_optimize))

        self.restore_state()
        # 프로그램 시작 알림 (Telegram & Discord)
        self.alert('[BOT 시작됨] 자동매매 봇이 초기화되었습니다.')

    def restore_state(self):
        try:
            with open(self.backup_file, 'r') as f:
                state = json.load(f)
                self.best_params.update(state.get('best_params', {}))
                self.trail_active = state.get('trail_active', False)
                self.max_profit_price = state.get('max_profit_price')
        except FileNotFoundError:
            pass

    def save_state(self):
        state = {
            'best_params': self.best_params,
            'trail_active': self.trail_active,
            'max_profit_price': self.max_profit_price
        }
        with open(self.backup_file, 'w') as f:
            json.dump(state, f)

    def alert(self, msg):
        print(msg)
        try:
            requests.post(
                f'https://api.telegram.org/bot{self.telegram_token}/sendMessage',
                data={'chat_id': self.telegram_chat_id, 'text': msg}
            )
            if self.discord_webhook:
                requests.post(self.discord_webhook, json={'content': msg})
        except Exception as e:
            print(f"Alert failed: {e}")

    def log_trade(self, event, side, entry, exit_price, amount):
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        with open('trade_log.csv', 'a') as f:
            f.write(f"{now},{event},{self.symbol},{side},{entry},{exit_price},{amount}\n")

    def get_indicators(self, df, a):
        df = df.copy()
        if 'ts' in df.columns:
            df = df.sort_values('ts')
        df = df.dropna(subset=['close'])
        rsi = RSIIndicator(df['close'], window=14).rsi()
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        macd = MACD(df['close']).macd_diff()

        print(a)

        if df['close'].nunique() == 1:
            print("⚠️ close 값이 전부 동일합니다. RSI가 고정됩니다.")
        else:
            print("✅ close 값에 변동이 있습니다. RSI가 제대로 계산됩니다.")
        print("RSI 마지막 값:", rsi.iloc[-1])

        return rsi, macd, bb.bollinger_lband(), bb.bollinger_hband()

    def get_5m_indicators(self):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '5m', 200)
        if len(ohlcv) < 50:
            self.alert("[Data Warning] 5m data insufficient, skipping indicator calc.")
            return pd.Series(), pd.Series(), pd.Series(), pd.Series()
        df5 = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'close', 'v'])
        return self.get_indicators(df5,2)

    def get_position(self):
        info = self.exchange.fetch_balance(params={'type': 'future'})['info']['positions']
        for p in info:
            if p['symbol'] == self.symbol.replace('/', ''):
                amt = float(p['positionAmt'])
                entry = float(p['entryPrice'])
                lev = float(p['leverage'])
                side = 'long' if amt > 0 else 'short' if amt < 0 else None
                return side, entry, lev, abs(amt)
        return None, 0, 1, 0

    def size_by_volatility(self, df):
        vol_range = (df['h'].max() - df['l'].min()) / df['close'].iloc[-1]
        vol_std = df['close'].pct_change().rolling(14).std().iloc[-1]
        vol = max(vol_range, vol_std)
        balance = self.exchange.fetch_balance()['USDT']['free']
        price = df['close'].iloc[-1]
        if vol < 0.007:
            pct = 0.6
        elif vol < 0.015:
            pct = 0.3
        else:
            pct = 0.1
        return (balance * pct) / price

    def _safe_order(self, func, *args):
        for attempt in range(2):
            try:
                return func(*args)
            except Exception as e:
                self.alert(f"Order error: {e}, retrying...")
                time.sleep(1)
        self.alert("Order failed after retries.")
        return None

    def open_long(self, amount):
        order = self._safe_order(
            self.exchange.create_market_buy_order, self.symbol, amount
        )
        if order:
            price = float(order['fills'][0]['price'])
            self.alert(f"[진입] Long {amount:.4f} @ {price:.2f}")
            self.log_trade('enter_long', 'long', price, None, amount)

    def open_short(self, amount):
        order = self._safe_order(
            self.exchange.create_market_sell_order, self.symbol, amount
        )
        if order:
            price = float(order['fills'][0]['price'])
            self.alert(f"[진입] Short {amount:.4f} @ {price:.2f}")
            self.log_trade('enter_short', 'short', price, None, amount)

    def close_position(self, amount):
        side, entry, _, _ = self.get_position()
        if side == 'long':
            order = self._safe_order(
                self.exchange.create_market_sell_order, self.symbol, amount
            )
            tag = 'exit_long'
        elif side == 'short':
            order = self._safe_order(
                self.exchange.create_market_buy_order, self.symbol, amount
            )
            tag = 'exit_short'
        else:
            return
        if order:
            price = float(order['fills'][0]['price'])
            self.alert(f"[청산] {side.capitalize()} {amount:.4f} @ {price:.2f}")
            self.log_trade(tag, side, entry, price, amount)

    def detect_crash(self, df1, rsi1, macd1, side):
        prev_price = df1['close'].iloc[-6]
        return (
            side and
            (df1['close'].iloc[-1] - prev_price) / prev_price <= -0.015 and
            (rsi1.iloc[-1] - rsi1.iloc[-6]) > 12 and
            macd1.iloc[-1] < 0
        )

    def detect_spike(self, df1, rsi1, macd1, side):
        prev2 = df1['close'].iloc[-2]
        return (
            side and
            (df1['close'].iloc[-1] - prev2) / prev2 >= 0.02 and
            rsi1.iloc[-1] > self.best_params['rsi_short'] and
            macd1.iloc[-2] < 0 and
            macd1.iloc[-1] > 0
        )

    def trade(self):
        try:
            ohlcv1 = self.exchange.fetch_ohlcv(self.symbol, '1m', 100)
            df1 = pd.DataFrame(
                ohlcv1,
                columns=['ts', 'o', 'h', 'l', 'close', 'v']
            )
            rsi1, macd1, bb_l, bb_h = self.get_indicators(df1,1)
            rsi5, macd5, _, _ = self.get_5m_indicators()
            side, entry, _, amt = self.get_position()
            price = df1['close'].iloc[-1]

            # 초기 RSI 출력
            if not hasattr(self, '_started'):
                print(f"Initial RSI: {rsi1.iloc[-1]:.2f}")
                self._started = True

            # Crash 처리
            if self.detect_crash(df1, rsi1, macd1, side):
                self.close_position(amt)
                size = self.size_by_volatility(df1)
                self.open_short(size)
                return

            # Spike 처리
            if self.detect_spike(df1, rsi1, macd1, side):
                self.close_position(amt)
                return

            # 신규 진입
            if not side:
                size = self.size_by_volatility(df1)
                if rsi1.iloc[-1] < 20 and macd1.iloc[-1] > 0:
                    balance = self.exchange.fetch_balance()['USDT']['free']
                    size = balance / price
                if (
                    rsi1.iloc[-1] < self.best_params['rsi_long'] and
                    price <= bb_l.iloc[-1] and
                    macd1.iloc[-1] > 0 and
                    rsi5.iloc[-1] < self.best_params['rsi_long'] and
                    macd5.iloc[-1] > 0
                ):
                    self.open_long(size)
                    return
                if (
                    rsi1.iloc[-1] > self.best_params['rsi_short'] and
                    price >= bb_h.iloc[-1] and
                    macd1.iloc[-1] < 0 and
                    rsi5.iloc[-1] > self.best_params['rsi_short'] and
                    macd5.iloc[-1] < 0
                ):
                    self.open_short(size)
                    return

            # 포지션 관리
            if side:
                pnl = (price - entry) / entry * (1 if side == 'long' else -1)
                if pnl <= -0.02:
                    self.close_position(amt * 0.3)
                elif pnl <= -0.04:
                    self.close_position(amt * 0.7)
                elif pnl <= -0.06:
                    self.close_position(amt)
                    return
                if not self.trail_active and pnl >= self.best_params['take_profit_pct']:
                    self.close_position(amt * 0.5)
                    self.trail_active = True
                    self.max_profit_price = price
                if self.trail_active:
                    if price > self.max_profit_price:
                        self.max_profit_price = price
                    elif price <= self.max_profit_price * (1 - 0.02):
                        _, _, _, cur_amt = self.get_position()
                        self.close_position(cur_amt)
                        self.trail_active = False
        except Exception as e:
            self.alert(f"Trade error: {e}")

    def run_backtest(self, data, params):
        # 백테스트 로직 Placeholder
        return pd.Series(dtype=float)

    def optimize_parameters(self):
        df = pd.read_csv(
            self.historical_csv,
            names=['ts', 'o', 'h', 'l', 'close', 'v']
        )
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        best_sharpe = -np.inf
        best_params = None
        for rl in [20, 25, 30]:
            for rs in [70, 75, 80]:
                for sl in [0.02, 0.03, 0.04]:
                    for tp in [0.1, 0.15]:
                        self.best_params.update({
                            'rsi_long': rl,
                            'rsi_short': rs,
                            'stop_loss_pct': sl,
                            'take_profit_pct': tp
                        })
                        rets = self.run_backtest(df, self.best_params)
                        if rets.empty:
                            self.alert(f"[Opt] No returns for params {self.best_params}")
                            continue
                        sharpe = rets.mean() / rets.std() * np.sqrt(len(rets))
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = self.best_params.copy()
        if best_params:
            self.best_params = best_params
            self.save_state()
            self.alert(f"[최적화] New params: {self.best_params}")

    def cmd_status(self, update, context):
        side, entry, lev, _ = self.get_position()
        price = self.exchange.fetch_ticker(self.symbol)['last']
        pr = (price - entry) / entry * (1 if side == 'long' else -1) if side else 0
        apr = pr * lev
        msg = f"Pos:{side}@{entry:.2f}|P&L:{apr:.2%}|Params:{self.best_params}"
        update.message.reply_text(msg)

    def cmd_optimize(self, update, context):
        self.optimize_parameters()
        update.message.reply_text('Optimization done')

    def start(self):
        # Telegram polling 및 메인 루프 시작
        self.application.run_polling()
        last_opt = None
        loop_count = 0
        while True:
            try:
                now = datetime.now()
                if (
                    now.weekday() == 0 and now.hour == 1 and now.minute == 0
                    and last_opt != now.date()
                ):
                    self.optimize_parameters()
                    last_opt = now.date()
                self.trade()
                loop_count += 1
                if loop_count % 60 == 0:
                    self.alert('[Health] Bot is running')
            except Exception as e:
                self.alert(f"Main loop error: {e}")
            time.sleep(60)


if __name__ == '__main__':
    bot = TradingBot()
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.alert('[BOT 종료됨] 자동매매 봇이 중단되었습니다.')
        bot.save_state()