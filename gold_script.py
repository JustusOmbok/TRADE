import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import pickle
import datetime

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

# Function to fetch latest data from MetaTrader5
def fetch_latest_data(symbol, timeframe, num_bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch data for {symbol}. Rates are None or empty.")
        return None  # Explicitly return None if there's an issue
    df = pd.DataFrame(rates)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')

    # Ensure volume is in the DataFrame
    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    elif 'real_volume' in df.columns:
        df['volume'] = df['real_volume']
    else:
        print("Volume not found")  # Default if volume is not available

    return df

# Main feature creation function with all updates
def create_features(df):
    # Existing indicators
    df['RSI'] = compute_rsi(df['close'], window=14)
    df['ATR'] = compute_atr(df['high'], df['low'], df['close'], window=14)
    df['rolling_std'] = df['close'].rolling(window=20).std()
    df['slowk'], df['slowd'] = compute_stochastic(df['high'], df['low'], df['close'])
    df['ROC'] = df['close'].pct_change(periods=12)
    df['CCI'] = compute_cci(df['high'], df['low'], df['close'], window=20)
    df['Williams_%R'] = compute_williams_r(df['high'], df['low'], df['close'], window=14)
    df['MFI'] = compute_mfi(df['high'], df['low'], df['close'], df['volume'], window=14)
    
    # New indicators
    # Keltner Channels
    df['keltner_upper'], df['keltner_lower'] = compute_keltner_channels(df['close'], df['high'], df['low'])

    # Divergence Detection (between Price and RSI)
    df['price_rsi_divergence'] = detect_divergence(df['close'], df['RSI'], lookback=5)

    # Heikin-Ashi Candlesticks
    df['ha_open'], df['ha_high'], df['ha_low'], df['ha_close'] = compute_heikin_ashi(df)

    df.dropna(inplace=True)
    return df

# Place a buy trade
def place_buy_trade(symbol, entry_price, stop_loss, take_profit, lotSize):

    # Place the initial buy order with the first take profit level
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lotSize,  # Divide the lot size into three parts
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

# Place a sell trade
def place_sell_trade(symbol, entry_price, stop_loss, take_profit, lotSize):

    # Place the initial sell order with the first take profit level
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lotSize,  # Divide the lot size into three parts
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

def load_modelA_and_scalerA():
    modelA = keras.models.load_model('GRU_xauusd_model.keras')
    with open('GRU_class_scaler.pkl', 'rb') as f:
        scalerA = pickle.load(f)
    # print(f"Scaler was fitted on features: {scaler.mean_}")  # Inspect the scaler's fitted features
    return modelA, scalerA

# Check if any active trades are open
def has_active_trade(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return len(positions) > 0

def wait_for_next_execution():
    while True:
        now = datetime.datetime.now()
        minute = now.minute
        second = now.second

        # Check if the time is one of the desired minute marks (00:01, 15:01, 30:01, 45:01)
        if (8 <= now.hour < 19) and (minute == 0 or minute == 15 or minute == 30 or minute == 45) and (second in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
            print(f"Starting the next trading cycle at {now.strftime('%H:%M:%S')} UTC")
            break
        else:
            # Sleep for 10 seconds before checking the time again
            time.sleep(10)

# Main trading function
def main_trading_loop():
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M15
    modelA, scalerA = load_modelA_and_scalerA()

    while True:
        order_id = None  # Initialize order_id to None

        if has_active_trade(symbol):
            print("An active trade is already open. Waiting for 15 minutes...")
            time.sleep(900)  # Sleep for 15 minutes
            continue
        wait_for_next_execution()
        df = fetch_latest_data(symbol, timeframe, num_bars=500)
        df = create_features(df)
        df.set_index('time', inplace=True)
        X_latestA = df.drop(columns=['tick_volume', 'real_volume', 'volume', 'spread', 'high', 'open', 'close', 'low']).select_dtypes(include=[np.number]).values[-1].reshape(1, -1)

        X_latestA =scalerA.transform(X_latestA)
        X_latestA = X_latestA.reshape(1, X_latestA.shape[1], 1)

        prediction_probs = modelA.predict(X_latestA)
        predictionA = np.argmax(prediction_probs, axis=-1)[0]
        confidence = prediction_probs[0][predictionA]

        if confidence < 0.7:
            print(f"Prediction confidence ({confidence:.2f}) is below the threshold. No trade will be placed.")
            continue
        print(f"Predicted action: {predictionA}")

        if predictionA == 1:  # Buy
            entry_price = df['close'].iloc[-1]  # Use the ask price for a buy trade
            stop_loss = entry_price - 3.00
            take_profit = entry_price + 3.00
            lotSize = 0.01
            order_id = place_buy_trade(symbol, entry_price, stop_loss, take_profit, lotSize)
            if order_id:
                print(f"Buy trade placed: Order ID = {order_id}")

        elif predictionA == 2:  # Sell
            entry_price = df['close'].iloc[-1]  # Use the bid price for a sell trade
            stop_loss = entry_price + 3.00
            take_profit = entry_price - 3.00
            lotSize = 0.01
            order_id = place_sell_trade(symbol, entry_price, stop_loss, take_profit, lotSize)
            if order_id:
                print(f"Sell trade placed: Order ID = {order_id}")

        # If no trade was placed, wait for 15 minutes
        if order_id is None:
            print("No favorable trading condition. Waiting for 30 minutes...")
            time.sleep(900)  # Sleep for 15 minutes
            continue  # Go back to the start of the loop

        # Wait for the trade to end
        if order_id:
            while True:
                # Check if the trade has ended
                if not has_active_trade(symbol):
                    print(f"Trade with Order ID {order_id} has ended.")
                    wait_for_next_execution()
                    break  # Exit the inner loop if all trades have ended
                time.sleep(60)  # Check every 60 seconds for price updates

if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()

    main_trading_loop()
    mt5.shutdown()
