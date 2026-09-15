import sqlite3
import pandas as pd
from forecast import train_tree_model
from sequence_model import train_lstm

with sqlite3.connect("retailiq.db") as con:
    weekly = pd.read_sql(
        "SELECT * FROM fact_weekly_sales",
        con
    )

_, mape, rmse = train_tree_model(weekly)
print(f"Tree model: MAPE={mape:.4f}, RMSE={rmse:.4f}")

train_lstm(weekly, epochs=5)
print("LSTM sequence model trained.")
