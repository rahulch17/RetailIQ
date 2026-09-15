from pathlib import Path
import numpy as np
import pandas as pd

ARTIFACT = Path("artifacts/lstm_model.keras")
METRICS = Path("artifacts/sequence_metrics.csv")

BASE_FEATURES = ["quantity", "avg_price", "promotion_rate"]


def make_sequences(group, lookback=8):
    g = group.sort_values("week_start").copy()
    values = g[BASE_FEATURES].astype(float).values
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i, 0])
    return np.asarray(X), np.asarray(y)


def train_sequence_model(weekly, lookback=8):
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except Exception as e:
        raise RuntimeError("TensorFlow is required for the LSTM comparison.") from e

    parts = []
    for _, g in weekly.groupby(["product_id", "store_id"]):
        if len(g) >= lookback + 10:
            X, y = make_sequences(g, lookback)
            if len(X):
                parts.append((X, y))

    if not parts:
        raise ValueError("Not enough data for LSTM sequences.")

    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])

    cut = int(len(X) * 0.8)
    Xtr, Xv, ytr, yv = X[:cut], X[cut:], y[:cut], y[cut:]

    model = Sequential([
        LSTM(32, input_shape=(lookback, len(BASE_FEATURES))),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=5, batch_size=64, verbose=0)

    pred = np.maximum(model.predict(Xv, verbose=0).ravel(), 0)
    rmse = float(np.sqrt(np.mean((yv - pred) ** 2)))
    mask = np.abs(yv) >= 1.0
    mape = float(np.mean(np.abs((yv[mask] - pred[mask]) / yv[mask])) * 100) if mask.any() else np.nan

    ARTIFACT.parent.mkdir(exist_ok=True)
    model.save(ARTIFACT)
    pd.DataFrame([{"model":"LSTM","rmse":rmse,"mape":mape,"lookback":lookback}]).to_csv(METRICS, index=False)
    print(f"LSTM RMSE: {rmse:.2f}")
    print(f"LSTM MAPE (non-zero actuals): {mape:.2f}%")
    return model
