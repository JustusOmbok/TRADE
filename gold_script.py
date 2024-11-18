import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import pickle
import datetime

class TradingBot:
    def __init__(self, symbol, timeframe, num_bars=500):
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars
        self.modelA, self.scalerA = self.load_modelA_and_scalerA()

    def fetch_latest_data(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.num_bars)
        if rates is None or len(rates) == 0:
            print(f"Failed to fetch data for {self.symbol}. Rates are None or empty.")
            return None
        df = pd.DataFrame(rates)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        if 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
        elif 'real_volume' in df.columns:
            df['volume'] = df['real_volume']
        else:
            print("Volume not found")
        return df

    def create_features(self, df):
        df['RSI'] = compute_rsi(df['close'], window=14)
        df['ATR'] = compute_atr(df['high'], df['low'], df['close'], window=14)
        df['rolling_std'] = df['close'].rolling(window=20).std()
        df['slowk'], df['slowd'] = compute_stochastic(df['high'], df['low'], df['close'])
        df['ROC'] = df['close'].pct_change(periods=12)
        df['CCI'] = compute_cci(df['high'], df['low'], df['close'], window=20)
        df['Williams_%R'] = compute_williams_r(df['high'], df['low'], df['close'], window=14)
        df['MFI'] = compute_mfi(df['high'], df['low'], df['close'], df['volume'], window=14)
        df['keltner_upper'], df['keltner_lower'] = compute_keltner_channels(df['close'], df['high'], df['low'])
        df['price_rsi_divergence'] = detect_divergence(df['close'], df['RSI'], lookback=5)
        df['ha_open'], df['ha_high'], df['ha_low'], df['ha_close'] = compute_heikin_ashi(df)
        df.dropna(inplace=True)
        return df

    def place_buy_trade(self, entry_price, stop_loss, take_profit, lotSize):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": "Buy trade TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place buy order: {result.comment}")
            return None
        return [result.order]

    def place_sell_trade(self, entry_price, stop_loss, take_profit, lotSize):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": "Sell trade TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place sell order: {result.comment}")
            return None
        return [result.order]

    def load_modelA_and_scalerA(self):
        modelA = keras.models.load_model('GRU_xauusd_model.keras')
        with open('GRU_class_scaler.pkl', 'rb') as f:
            scalerA = pickle.load(f)
        return modelA, scalerA

    def has_active_trade(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) > 0

    def wait_for_next_execution(self):
        while True:
            now = datetime.datetime.now()
            minute = now.minute
            second = now.second
            if (8 <= now.hour < 19) and (minute in [0, 15, 30, 45]) and (second in range(21)):
                print(f"Starting the next trading cycle at {now.strftime('%H:%M:%S')} UTC")
                break
            else:
                time.sleep(10)

# Helper functions for indicators

def compute_keltner_channels(close, high, low, ema_window=20, atr_window=10, atr_multiplier=2):
    ema = close.ewm(span=ema_window, adjust=False).mean()
    atr = compute_atr(high, low, close, window=atr_window)
    upper_band = ema + (atr_multiplier * atr)
    lower_band = ema - (atr_multiplier * atr)
    return upper_band, lower_band

def detect_divergence(price, indicator, lookback=5):
    price_highs = price.rolling(window=lookback).max()
    price_lows = price.rolling(window=lookback).min()
    indicator_highs = indicator.rolling(window=lookback).max()
    indicator_lows = indicator.rolling(window=lookback).min()
    divergence = np.where((price > price_highs.shift(1)) & (indicator < indicator_highs.shift(1)), 1,
                          np.where((price < price_lows.shift(1)) & (indicator > indicator_lows.shift(1)), -1, 0))
    return pd.Series(divergence, index=price.index)

def compute_heikin_ashi(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close

def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_stochastic(high, low, close, window=14):
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    slowk = 100 * (close - lowest_low) / (highest_high - lowest_low)
    slowd = slowk.rolling(window=3).mean()
    return slowk, slowd

def compute_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci

def compute_williams_r(high, low, close, window=14):
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def compute_mfi(high, low, close, volume, window=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()
    money_flow_index = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    return money_flow_index
