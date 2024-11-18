import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import datetime

# Function to fetch historical data from MetaTrader5
def fetch_data(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        print(f"No data found for symbol {symbol} in the given date range.")
        return pd.DataFrame()

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

    # Define multi-class target variable
    df['target'] = np.where((df['close'].shift(2) < df['close'].shift(1)) & (df['close'] > df['close'].shift(1)), 1,
                            np.where((df['close'].shift(2) > df['close'].shift(1)) & (df['close'] < df['close'].shift(1)), 2,
                                     0))  # No favorable trade
    df.dropna(inplace=True)
    return df

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

# CNN-GRU model training function with hyperparameter tuning
def train_cnn_gru_model(X_train, y_train, X_test, y_test, scaler):
    model = keras.Sequential()
    
    # Adding CNN layers to capture spatial relationships
    model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    
    # Adding GRU layers to capture sequential dependencies
    model.add(keras.layers.GRU(50, return_sequences=True))  # GRU layer
    model.add(keras.layers.Dropout(0.2))  # Dropout to avoid overfitting
    model.add(keras.layers.GRU(50, return_sequences=False))  # GRU layer
    model.add(keras.layers.Dropout(0.2))  # Dropout layer
    
    # Fully connected output layer
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))  # Output layer for multi-class classification
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks for early stopping and learning rate adjustment
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Fit the model with the callbacks
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
              callbacks=[lr_scheduler, early_stopping], verbose=2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Loss: {loss}")

    # Predictions and detailed classification report
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred, target_names=['No Trade', 'Buy', 'Sell']))

    # Save the model and scaler in the recommended format
    model.save('GRU_xauusd_model.keras')  # Updated format
    with open('GRU_class_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# --- Main Script ---
if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()

    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M15
    start = datetime.datetime(2022, 1, 10)
    end = datetime.datetime.now()

    df = fetch_data(symbol, timeframe, start, end)

    if df.isna().sum().any():
        print("Missing data detected. Handling NaN values...")
        df.ffill(inplace=True)  # Forward-fill any missing data points

    # Check if 'volume' is in the fetched DataFrame
    if 'volume' in df.columns:
        print("Volume data included.")
    else:
        print("Volume data not available.")

    if df.empty:
        print("No data fetched. Exiting script.")
        mt5.shutdown()
        quit()
        # Filter data based on 'hour_of_day'
    
    # Feature engineering first
    df = create_features(df)  # Adjust look-back period here

    print("Columns in df_balanced:", df.columns.tolist())

    print(df.shape)  # Check shape before setting index
    df.set_index('time', inplace=True)

    # Prepare features and target
    X = df.drop(columns=['tick_volume', 'real_volume', 'volume', 'spread', 'high', 'open', 'close', 'low', 'target']).select_dtypes(include=[np.number]).values  # Only keep numeric columns
    y = df['target'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # Print feature shapes for debugging
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train the LSTM mode
    train_cnn_gru_model(X_train, y_train, X_test, y_test, scaler)

    mt5.shutdown()