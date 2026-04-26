# I Tried to Predict Bitcoin's Price Using a Neural Network — Here's What Happened

*A step-by-step walkthrough of building a GRU-based time series forecasting model on real Coinbase and Bitstamp data — from raw CSVs to a $26 average error.*

---

![Bitcoin price chart with GRU forecast overlay — predicted line closely tracking actual BTC/USD price across the Sep–Jan 2018/2019 test period]

> **TL;DR:** Using 24 hours of 1-minute BTC/USD candles as input, a GRU model predicted the close price 1 hour into the future with a mean absolute error of **$26** and a MAPE of **0.56%** on the 2018 bear market test set. The full code is on GitHub.

---

## Everyone Wants to Know How to Make Money with Bitcoin

Let's be honest — the reason most people get interested in forecasting BTC is the same reason they get interested in BTC at all: the possibility of profit. But before you can trade on a prediction, you need to actually make one. And making a reliable prediction on a financial time series is a lot harder than it looks.

This post walks through exactly how I built a machine learning model to predict the BTC/USD close price 1 hour into the future, using real historical data from Coinbase and Bitstamp. Along the way I will cover:

- What time series forecasting actually means
- Why the raw data needs serious cleaning before a model can touch it
- How to build a memory-efficient data pipeline with `tf.data`
- The GRU architecture I chose and why
- The results — including what a 0.56% MAPE actually means in practice

---

## Part 1 — What Is Time Series Forecasting?

A **time series** is any sequence of measurements taken at regular time intervals: hourly temperature readings, monthly sales figures, or in our case, one-minute BTC/USD candles.

**Forecasting** means using the history of that sequence to predict future values. The core assumption is that the past contains signal about the future — that patterns like "price tends to recover after a sharp drop in volume" repeat in ways a model can learn.

The classic approach is to define a **look-back window** and a **prediction horizon**:

```
[t-1439, t-1438, ..., t-1, t]  →  predict  →  [t+60]
        ↑ 24 hours of history                      ↑ 1 hour ahead
```

Every model input is a window of 1,440 one-minute candles (24 hours). Every model output is a single number: the predicted close price 60 minutes after the last candle in the window.

The key rule that makes time series different from standard ML: **you cannot shuffle the data**. If you randomly split your dataset into train and test, a training sample from Thursday will sit next to a test sample from Tuesday of the same week. The model has already "seen" the market context around your test points — that is data leakage, and your results will be meaninglessly optimistic.

---

## Part 2 — The Data and Why It Needed Serious Cleaning

The raw datasets come from Coinbase and Bitstamp. Each row represents a 60-second window:

| Column | Description |
|--------|-------------|
| `Timestamp` | Unix epoch (seconds) |
| `Open` | Opening price in USD |
| `High` | Highest price in the window |
| `Low` | Lowest price in the window |
| `Close` | Closing price in USD |
| `Volume_(BTC)` | BTC volume transacted |
| `Volume_(Currency)` | USD volume transacted |
| `Weighted_Price` | Volume-weighted average price (VWAP) |

Sounds clean. It was not.

### Problem 1 — Two types of missing data

Inspecting the raw Coinbase CSV revealed ~**120,000 rows** with entirely missing OHLCV values. These are not corrupted rows — they represent minutes where no trade occurred on the exchange. The last known price is still the valid market price; the volume is genuinely zero.

But there was a second, invisible type of missing data: **entire minutes absent from the CSV altogether**. A gap from timestamp `1483228800` to `1483236000` does not appear as NaN rows — it simply does not exist. A naive `ffill()` would never catch this.

The fix: reindex the entire dataset against a complete 60-second grid before filling anything.

```python
full_idx = np.arange(
    int(df.index.min()),
    int(df.index.max()) + 60,
    60, dtype=np.int64
)
df = df.reindex(full_idx)          # inserts absent minutes as NaN
df[price_cols]  = df[price_cols].ffill()
df[volume_cols] = df[volume_cols].fillna(0.0)
```

### Problem 2 — 2014–2016 is almost useless

The dataset starts in December 2014, but BTC trading on Coinbase before 2017 was extremely thin. The longest consecutive gap was **nearly 38 hours**. Forward-filling that gap produces a perfectly flat line — the same price repeated 2,280 times. A model trained on that would learn that "flat price + zero volume for days" is a real market pattern.

**Decision: trim everything before 2017-01-01.**

### Problem 3 — Most features are redundant

Running a full correlation matrix across all columns revealed something immediately obvious but easy to overlook:

Open, High, Low, Close, and Weighted_Price all correlate at **> 0.9999** with each other. They are essentially the same number. Keeping all five wastes model capacity without adding any signal.

`Volume_(Currency)` is also redundant — it is approximately `Volume_(BTC) × Close`, so it contains no independent information.

**Final feature set: 6 columns.**

```python
FEATURES = ['Close', 'Volume_BTC_log', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
```

### Problem 4 — Volume is log-normal

Raw `Volume_(BTC)` is extremely right-skewed. After `MinMaxScaler`, almost every row sits near zero and a handful of spikes consume the top of the [0, 1] range. The model cannot distinguish a quiet minute from a moderately active one.

Fix: `log1p(volume)` before scaling, which compresses the distribution into something approximately normal.

### Problem 5 — Time needs cyclic encoding

Encoding hour-of-day as an integer (0–23) tells the model that midnight (0) and 11pm (23) are 23 units apart. They are actually 1 hour apart. The fix is to map each periodic feature onto the unit circle:

```python
df['hour_sin'] = np.sin(2 * np.pi * dt_idx.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * dt_idx.hour / 24)
df['dow_sin']  = np.sin(2 * np.pi * dt_idx.dayofweek / 7)
df['dow_cos']  = np.cos(2 * np.pi * dt_idx.dayofweek / 7)
```

Now midnight and 11pm are neighbours on the circle, as they should be.

### The split — and why scaling comes after

The dataset (2017–early 2019) was split chronologically:

| Set | Date range | Ratio |
|-----|-----------|-------|
| Train | Jan 2017 → May 2018 | 70% |
| Val | Jun 2018 → Aug 2018 | 13% |
| Test | Sep 2018 → Jan 2019 | 17% |

Critically, `MinMaxScaler` was fitted **only on the training set** before transforming the others. Fitting on the full dataset would let the scaler's `max_` be set by the late-2017 ATH of ~$14,000. A training sample from early 2017 at ~$1,000 would then be scaled to ~0.07 — a value calibrated against future price levels the model should not know about. That is data leakage.

---

## Part 3 — Building the `tf.data` Pipeline

The naive approach to building sliding windows is a Python loop:

```python
for i in range(n - window_size - horizon + 1):
    X.append(data[i : i + window_size])
    y.append(data[i + window_size + horizon - 1, 0])
```

For ~733,000 windows of shape `(1440, 6)`, this allocates roughly **25 GB of RAM** before training even starts. On Kaggle's free tier this crashes the kernel immediately.

The solution is to never materialise the window arrays at all. Instead, store only the flat scaled time series (~25 MB) and let `tf.data` slice windows lazily, one batch at a time:

```python
def make_window_dataset(series, window_size, horizon, batch_size, stride=1, shuffle=False):
    T         = tf.cast(tf.shape(series)[0], tf.int64)
    ws        = tf.cast(window_size, tf.int64)
    hz        = tf.cast(horizon,     tf.int64)
    st        = tf.cast(stride,      tf.int64)
    n_windows = (T - ws - hz) // st + 1

    def get_window(idx):
        i = idx * st
        X = series[i : i + window_size]
        y = series[i + window_size + horizon - 1, 0]
        return X, y

    ds = tf.data.Dataset.range(n_windows)
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000, seed=42)
    ds = ds.map(get_window, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
```

`Dataset.range(n_windows)` yields integer start indices — no data is copied. The `map()` call slices `(X, y)` pairs on demand. Peak RAM stays under **1 GB** regardless of dataset size.

A `stride` parameter controls how densely the windows are sampled. `stride=10` uses every 10th possible start index, reducing steps-per-epoch by 10× while still training on complete 24-hour sequences. This brought training time from ~8 hours to ~20 minutes on a Kaggle T4 GPU.

---

## Part 4 — The GRU Model

### Why GRU and not LSTM?

| Architecture | Gates | Parameters | Long-range memory |
|:---:|:---:|:---:|:---:|
| Vanilla RNN | 0 | Fewest | Poor (vanishing gradients) |
| LSTM | 3 | Most | Excellent |
| **GRU** | **2** | **~25% fewer** | **Good** |

GRU (Gated Recurrent Unit, Cho et al. 2014) has two gates — reset and update — compared to LSTM's three. This means roughly 25% fewer parameters with comparable performance on sequential data. For a 1,440-step input sequence, that difference in training speed is meaningful.

### Architecture

```
Input  (1440, 6)
└─ GRU(64, return_sequences=False)
└─ Dropout(0.2)
└─ Dense(1, linear)
```

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1440, 6)),
    keras.layers.GRU(64, return_sequences=False, name='gru_1'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='linear'),
], name='GRU_BTC_Fast')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae'],
)
```

A single GRU layer is sufficient here. The task is compressing a 1,440-step sequence into a single price estimate — one recurrent layer already does that. Stacking a second layer adds parameters and training time without meaningfully improving accuracy at this scale.

**Loss function: MSE.** Mean squared error penalises large prediction errors quadratically, which matters for financial forecasting where a catastrophically wrong prediction is worse than two moderately wrong ones.

### Training setup

Three callbacks kept training efficient and safe:

- `EarlyStopping(patience=5)` — halted training when validation loss stopped improving, typically around epoch 12–18
- `ReduceLROnPlateau(factor=0.5, patience=3)` — halved the learning rate when the loss plateaued
- `ModelCheckpoint` — saved the best weights regardless of when early stopping fired

---

## Part 5 — Results

### Training curves

The model converged cleanly. Both train and validation loss decreased together without diverging, and the learning rate reduction events are visible as small drops in the loss curve around epochs 8 and 13.

### Test set performance

| Metric | Value |
|--------|-------|
| MSE (scaled) | 0.000005 |
| MAE (scaled) | 0.001324 |
| RMSE (USD) | $42 |
| MAE (USD) | $26 |
| MAPE | 0.56% |
| Residual mean | $1.3 |
| Residual std | $42.4 |

**How to read these numbers:**

The test set covers September 2018 to January 2019 — the heart of the BTC bear market, when price fell from ~$6,500 to ~$3,200. This is one of the hardest periods to predict because the trend is strongly directional and volatility is high.

A MAE of **$26** on prices ranging from $3,200 to $6,500 translates to roughly 0.4–0.8% error on the actual price level. The MAPE of **0.56%** confirms this — on average, the model's 1-hour-ahead prediction is wrong by less than 1% of the real price.

The residual mean of **$1.3** is the most encouraging number. It means the model has almost no systematic bias — it is not consistently over- or under-predicting. The errors are centred near zero.

The gap between RMSE ($42) and MAE ($26) reveals the tail: most predictions are close, but a handful of large misses (likely during sudden price drops) inflate the RMSE. This is normal for volatile financial data.

### What does 0.56% MAPE actually mean?

BTC typically moves 1–3% in a normal hour. A model error of 0.56% means the uncertainty in the prediction is smaller than the signal it is trying to predict — at least on average. That is a meaningful result, not just a statistical artefact.

However, one important check: does the model genuinely learn patterns, or is it mostly learning persistence (predicting that the price will be close to whatever it is now)? Adding a naive baseline — predicting `close[t+60] = close[t]` — would tell us exactly how much of the 0.56% MAPE comes from "it will probably stay near where it is" versus genuine learned signal.

---

## Part 6 — Conclusion and Thoughts on Forecasting BTC

### What I learned

The data pipeline was harder than the model. Getting the preprocessing right — reindexing, the two types of missing data, the correct split order, the log transform — took more thought than choosing the GRU architecture. This is almost always true in real ML projects: the model is 20% of the work.

The `tf.data` lazy windowing approach was the biggest practical discovery. The naive numpy approach that crashes on 25 GB arrays is what most tutorials show. The lazy pipeline pattern works at any scale and is worth learning regardless of the specific task.

### Thoughts on actually trading with this

I would not trade real money based on these results alone. A 0.56% MAPE is impressive as a research result, but it does not account for:

- **Transaction fees** — BTC trading fees on most exchanges are 0.1–0.5% per trade, which can eat the entire prediction edge
- **Slippage** — actually executing at the predicted price is not guaranteed, especially in fast-moving markets
- **Regime change** — the model was trained on 2017–2018 data; it has no way to handle fundamentally new market conditions
- **The persistence baseline** — we do not yet know how much of the model's performance is genuinely predictive versus "the price tends not to move dramatically in one hour"

What this model is legitimately useful for is as a starting point: a clean pipeline, a working architecture, and a solid preprocessing methodology that can be extended with additional features, longer look-back windows, or ensemble approaches.

### What I would try next

- Add the persistence baseline test to quantify genuine model contribution
- Include on-chain features (transaction volume, active addresses) as additional inputs
- Experiment with a Transformer architecture, which handles long sequences differently from GRUs
- Train on multiple assets simultaneously to see if cross-asset patterns improve BTC predictions

### Code

The full preprocessing and training code is available on GitHub:

**[github.com/YOUR_USERNAME/holbertonschool-machine_learning/tree/main/supervised_learning/time_series](https://github.com/YOUR_USERNAME/holbertonschool-machine_learning/tree/main/supervised_learning/time_series)**

---

*Thanks for reading. If you found this useful, the most helpful thing you can do is run the baseline test and tell me how it goes.*

---

**Tags:** `Machine Learning` `Bitcoin` `Time Series` `Deep Learning` `Python` `TensorFlow` `GRU` `Data Science`
