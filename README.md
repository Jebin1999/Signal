

```markdown
 __Crypto Signal Predictor (ETH/USDT) using LSTM__ 

This project builds an LSTM model to predict short-term  prices using 1-minute OHLCV data via the `ccxt` library. The trained model is used for **live predictions** with confidence probability — ideal for crypto futures signal generation.

---



---

## What It Does

- Collects 1-minute OHLCV data of ETH/USDT from Binance via `ccxt`
- Trains an LSTM neural network for price prediction
- Evaluates the model (R², MAE, RMSE)
- Performs **live price prediction** and prints prediction probability
- Supports real-time crypto trading signal analysis

---

##  Requirements

Make sure to install:

```bash
pip install -r requirements.txt
````

### `requirements.txt`

```text
ccxt
pandas
numpy
matplotlib
scikit-learn
tensorflow
joblib
```

---

##  How to Run

### Step 1: Train the model

```bash
python scripts/Trainingmodel.py
```

This trains the LSTM model and saves:

* `models/ethusdt_model.h5`
* `models/ethusdt_scaler.pkl`

###  Step 2: Run live prediction

```bash
python scripts/LivePredictor.py
```

You’ll see:

```
📡 Live ETH/USDT Prediction
🟢 Price: 2925.73
🔮 Predicted: 2926.18
📈 Confidence: 93.2%
```

---

## 🇮🇪 Notes

* Timezone is aligned to **Europe/Dublin (Ireland)**.
* LSTM models are sensitive — ensure consistent scaling and windowing.

---

## 💡 TODO (Next Steps)

* [ ] Add email or Telegram alerts
* [ ] Integrate trading logic for automated bots
* [ ] Dockerize the project for deployment

---

## 📬 Author

Jebin Larosh — \[Data Scientist, Ireland 🇮🇪]

---

## ⚠️ Disclaimer

This is for educational/research purposes only. Not financial advice. Trade at your own risk.

```
