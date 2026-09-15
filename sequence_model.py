from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

class LSTMForecaster(nn.Module):
    def __init__(self, n_features=4, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def train_lstm(weekly, epochs=5, seq_len=8):
    x = weekly.copy()
    x["week_start"] = pd.to_datetime(x["week_start"])
    x = x.sort_values(["product_id", "store_id", "week_start"])

    X, y = [], []

    for _, g in x.groupby(["product_id", "store_id"]):
        g = g.reset_index(drop=True)
        if len(g) <= seq_len:
            continue

        q = g["quantity"].to_numpy(float)
        p = g["avg_price"].to_numpy(float)
        promo = g["promotion_rate"].to_numpy(float)

        for i in range(seq_len, len(g)):
            window = np.column_stack([
                q[i-seq_len:i],
                p[i-seq_len:i],
                promo[i-seq_len:i],
                np.arange(seq_len) / seq_len,
            ])
            X.append(window)
            y.append(q[i])

    if not X:
        raise ValueError("Not enough sequential history to train LSTM.")

    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.float32)

    split = int(len(X) * 0.8)
    loader = DataLoader(
        TensorDataset(X[:split], y[:split]),
        batch_size=256,
        shuffle=True
    )

    model = LSTMForecaster(X.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), ART / "lstm_model.pt")
    return model

if __name__ == "__main__":
    import sqlite3
    with sqlite3.connect("retailiq.db") as con:
        weekly = pd.read_sql(
            "SELECT * FROM fact_weekly_sales",
            con
        )
    train_lstm(weekly)
    print("LSTM model saved.")
