import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
from gold_script import TradingBot


# Main trading function
def main_trading_loop():
    symbol = "XAUUSD"
    bot = TradingBot(symbol='XAUUSD', timeframe=mt5.TIMEFRAME_M15, num_bars=500)
    modelA, scalerA = bot.load_modelA_and_scalerA()

    while True:
        order_id = None  # Initialize order_id to None

        if bot.has_active_trade():
            print("An active trade is already open. Waiting for 15 minutes...")
            time.sleep(900)  # Sleep for 15 minutes
            continue
        bot.wait_for_next_execution()
        df = bot.fetch_latest_data()
        df = bot.create_features(df)
        df.set_index('time', inplace=True)
        X_latestA = df.drop(columns=['tick_volume', 'real_volume', 'volume', 'spread', 'high', 'open', 'close', 'low']).select_dtypes(include=[np.number]).values[-1].reshape(1, -1)

        X_latestA =scalerA.transform(X_latestA)
        X_latestA = X_latestA.reshape(1, X_latestA.shape[1], 1)

        prediction_probs = modelA.predict(X_latestA)
        predictionA = np.argmax(prediction_probs, axis=-1)[0]
        confidence = prediction_probs[0][predictionA]

        if confidence < 0.7:
            print(f"Prediction confidence ({confidence:.2f}) is below the threshold. No trade will be placed.")
            time.sleep(900)  # Sleep for 15 minutes
            continue
        print(f"Predicted action: {predictionA}")

        if predictionA == 1:  # Buy
            entry_price = df['close'].iloc[-1]  # Use the ask price for a buy trade
            stop_loss = entry_price - 3.00
            take_profit = entry_price + 3.00
            lotSize = 0.01
            order_id = bot.place_buy_trade(symbol, entry_price, stop_loss, take_profit, lotSize)
            if order_id:
                print(f"Buy trade placed: Order ID = {order_id}")

        elif predictionA == 2:  # Sell
            entry_price = df['close'].iloc[-1]  # Use the bid price for a sell trade
            stop_loss = entry_price + 3.00
            take_profit = entry_price - 3.00
            lotSize = 0.01
            order_id = bot.place_sell_trade(symbol, entry_price, stop_loss, take_profit, lotSize)
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
                if not bot.has_active_trade():
                    print(f"Trade with Order ID {order_id} has ended.")
                    bot.wait_for_next_execution()
                    break  # Exit the inner loop if all trades have ended
                time.sleep(60)  # Check every 60 seconds for price updates

if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()

    main_trading_loop()
    mt5.shutdown()