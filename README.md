# BTC Price Forecasting with GRU

> Holberton School — `supervised_learning/time_series`

A full pipeline that trains a stacked GRU (Gated Recurrent Unit) network to predict the BTC/USD close price **1 hour into the future**, using the **past 24 hours** of 1-minute OHLCV candles from Coinbase and Bitstamp.

---
###TRY  THE DEMO HERE https://huggingface.co/spaces/Megi96/Bitcoin_Forecast
## Files

| File | Purpose |
|------|---------|
| `preprocess_data.py` | Cleans raw CSV(s), engineers features, splits data, scales, builds sliding windows, saves `.npz` + `.pkl` |
| `forecast_btc.py` | Loads preprocessed arrays, builds a GRU model with `tf.data`, trains, evaluates, saves model and plots |
| `README.md` | This file |

---

## Datasets

Download the raw 1-minute CSVs from Kaggle:

- **Coinbase** – `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`
- **Bitstamp** – `bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv`

Each row represents one 60-second window containing:

| Column | Description |
|--------|-------------|
| `Timestamp` | Unix epoch (seconds) at window start |
| `Open` | Opening price (USD) |
| `High` | Highest price in the window |
| `Low` | Lowest price in the window |
| `Close` | Closing price (USD) |
| `Volume_(BTC)` | BTC volume transacted |
| `Volume_(Currency)` | USD volume transacted |
| `Weighted_Price` | Volume-weighted average price (VWAP) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn tensorflow joblib matplotlib

# 2. Preprocess (both datasets, or just one)
python preprocess_data.py \
    --coinbase ./data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv \
    --bitstamp ./data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv \
    --out_dir  ./data

# 3. Train and evaluate
python forecast_btc.py \
    --data_dir ./data \
    --out_dir  ./outputs
```

Outputs are written to `./outputs/`:

```
outputs/
├── best_gru_btc.keras       # best checkpoint (by val_loss)
├── gru_btc_final.keras      # final saved model
├── training_curves.png      # loss & MAE learning curves
├── prediction_plot.png      # predicted vs actual BTC price
└── residuals_plot.png       # residual distribution & scatter
```

---

## Preprocessing Pipeline

The order of operations is strictly enforced because each step depends on the previous one:

```
1.  Load raw CSV(s) and sort by Timestamp
2.  Merge Coinbase + Bitstamp (Coinbase preferred where both have data)
3.  Trim to 2017-01-01                     ← BEFORE reindex (saves memory)
4.  Reindex to a complete 60-second grid   ← inserts absent minutes as NaN
5.  Forward-fill price columns             ← no-trade minute = last known price
    Zero-fill volume columns               ← no-trade minute = zero volume
6.  log1p(Volume_BTC)                      ← compress log-normal distribution
7.  Cyclic time encoding                   ← sin/cos of hour-of-day and day-of-week
8.  Drop redundant columns                 ← keep: Close, log-vol, 4 time features
9.  Chronological split                    ← BEFORE scaling (no data leakage)
       Train : Jan 2017 → May 2018  (~70%)
       Val   : Jun 2018 → Aug 2018  (~13%)
       Test  : Sep 2018 → Jan 2019  (~17%)
10. Fit MinMaxScaler on TRAIN ONLY
11. Transform train, val, test
12. Build sliding windows (vectorised numpy strides — fast)
       X shape : (N, 1440, 6)   — 24 h of 6 features
       y shape : (N,)           — Close price 60 min after the window ends
13. Save btc_windows.npz + btc_scaler.pkl
```

### Why these choices?

| Decision | Reason |
|----------|--------|
| **Trim 2014-2016** | Those years have gaps up to 38 hours; forward-filling them injects false "flat price" patterns that a model would learn as legitimate market signals |
| **Reindex before fill** | Without reindexing, whole minutes absent from the CSV remain undetected. A 1 440-row window would silently span more than 24 real hours |
| **Keep only `Close`** | Open/High/Low/Weighted_Price all correlate > 0.9999 with Close — redundant features waste capacity without improving predictions |
| **log1p(Volume)** | Raw BTC volume is heavily right-skewed; after MinMaxScaler almost every row would be near 0, making quiet vs active trading indistinguishable |
| **Cyclic time encoding** | Integer hour (0–23) makes midnight and 11 pm seem far apart (distance = 23). Sin/cos encoding keeps them neighbours on the unit circle |
| **Scale after split** | Fitting the scaler on the full dataset lets future price levels (e.g. the 2017 ATH of $14 000) leak into the training scale — a form of look-ahead bias |
| **Target = Close[t + 1440 + 60]** | Predicting the last row of the input window would be trivially easy and useless in production. The target must be genuinely 60 min in the future |

---

## Model Architecture

```
Input  (1440, 6)
├─ GRU(128, return_sequences=True)    ← all 1440 hidden states passed forward
├─ Dropout(0.2)
├─ GRU(64,  return_sequences=False)   ← single context vector
├─ Dropout(0.2)
├─ Dense(32, relu)
└─ Dense(1,  linear)                  ← predicted scaled Close price
```

**Why GRU?**

| Architecture | Gates | Parameters | Long-range memory |
|:---:|:---:|:---:|:---:|
| Vanilla RNN | 0 | Fewest | Poor (vanishing gradients) |
| LSTM | 3 | Most | Excellent |
| **GRU** | **2** | **~25% fewer than LSTM** | **Good** |

GRU matches LSTM's ability to handle long sequences while training significantly faster — a meaningful advantage for 1 440-step inputs.

**Loss:** MSE (required by project spec)  
**Optimizer:** Adam with `ReduceLROnPlateau` and `EarlyStopping`

---

## Key Improvement Over Original Notebook

The original notebook used a Python `for` loop to build sliding windows:

```python
# SLOW — O(N) Python iterations, causes timeouts on Kaggle for large datasets
for i in range(n - window_size - horizon + 1):
    X.append(data[i : i + window_size])
```

`preprocess_data.py` replaces this with **vectorised numpy stride tricks**, which builds all windows as a zero-copy memory view and is orders of magnitude faster:

```python
X = np.lib.stride_tricks.as_strided(
    data,
    shape=(n_windows, window_size, n_features),
    strides=(item_size, item_size, feat_size),
).copy().astype(np.float32)
```

---

## Dependencies

```
tensorflow >= 2.12
numpy
pandas
scikit-learn
joblib
matplotlib
```

---

## Repository

```
holbertonschool-machine_learning/
└── supervised_learning/
    └── time_series/
        ├── README.md
        ├── preprocess_data.py
        └── forecast_btc.py
```
