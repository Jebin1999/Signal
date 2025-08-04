import ccxt
import pandas as pd
import numpy as np
import time
import joblib
import os 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === Step 1: Fetch 2 years of historical 1h ETH/USDT data ===
def fetch_ohlcv(symbol="ETH/USDT", timeframe="1h", since_days=730):
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=since_days)).isoformat())
    all_data = []

    while since < exchange.milliseconds():
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not data:
                break
            all_data += data
            since = data[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df[["close"]]

df = fetch_ohlcv()
print("✅ Data fetched:", df.shape)
print(df.tail())

# === Step 2: Preprocess ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df["close"].values.reshape(-1, 1))

# Train-test split
window = 60
X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i - window:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Step 3: Build and train model ===
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(64))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# === Step 4: Evaluate ===
pred = model.predict(X_test)
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test_inv, pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, pred_inv)
r2 = r2_score(y_test_inv, pred_inv)

print("\n📊 Model Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Save the model and scaler

model.save("models/ethusdt_model.h5")
joblib.dump(scaler, "models/ethusdt_scaler.pkl")

print("✅ Model and scaler saved successfully.")
