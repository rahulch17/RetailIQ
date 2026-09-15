from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = Path("artifacts/tree_model.joblib")
METRICS_PATH = Path("artifacts/metrics.csv")

FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_4", "roll_8",
    "avg_price", "promotion_rate",
    "weekofyear", "month"
]


def make_features(df):
    x = df.copy()
    x["week_start"] = pd.to_datetime(x["week_start"])
    x = x.sort_values("week_start").reset_index(drop=True)
    for lag in [1, 2, 4, 8]:
        x[f"lag_{lag}"] = x["quantity"].shift(lag)
    x["roll_4"] = x["quantity"].shift(1).rolling(4).mean()
    x["roll_8"] = x["quantity"].shift(1).rolling(8).mean()
    x["weekofyear"] = x["week_start"].dt.isocalendar().week.astype(int)
    x["month"] = x["week_start"].dt.month
    return x


def forward_split(df, validation_fraction=0.2):
    df = df.sort_values("week_start").reset_index(drop=True)
    cut = max(1, int(len(df) * (1 - validation_fraction)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def safe_mape(y_true, y_pred, threshold=1.0):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.abs(y) >= threshold
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def train_tree_model(weekly):
    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    for c, default in [("avg_price", 1.0), ("promotion_rate", 0.0)]:
        if c not in weekly:
            weekly[c] = default
        weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(default)
    weekly["quantity"] = pd.to_numeric(weekly["quantity"], errors="coerce").fillna(0.0)

    train_parts, val_parts = [], []

    for _, g in weekly.groupby(["product_id", "store_id"]):
        if len(g) < 12:
            continue
        f = make_features(g).dropna(subset=FEATURES + ["quantity"])
        if len(f) < 12:
            continue
        tr, va = forward_split(f)
        if len(tr) >= 8 and len(va) >= 1:
            train_parts.append(tr)
            val_parts.append(va)

    if not train_parts:
        raise ValueError("No valid Store x Department series for training.")

    tr = pd.concat(train_parts, ignore_index=True)
    va = pd.concat(val_parts, ignore_index=True)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=0.1,
        random_state=42
    )
    model.fit(tr[FEATURES], tr["quantity"])

    pred = np.maximum(model.predict(va[FEATURES]), 0)
    mae = mean_absolute_error(va["quantity"], pred)
    rmse = float(np.sqrt(mean_squared_error(va["quantity"], pred)))
    mape = safe_mape(va["quantity"], pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = pd.DataFrame([{
        "model": "HistGradientBoostingRegressor",
        "mae": mae, "rmse": rmse, "mape": mape,
        "train_rows": len(tr), "validation_rows": len(va),
        "features": ",".join(FEATURES)
    }])
    metrics.to_csv(METRICS_PATH, index=False)

    print(f"Training rows: {len(tr):,}")
    print(f"Validation rows: {len(va):,}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (non-zero actuals): {mape:.2f}%")
    return model, mape, rmse


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run python train_models.py")
    model = joblib.load(MODEL_PATH)
    if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != FEATURES:
        raise ValueError("Saved model features do not match forecast.py. Retrain.")
    return model


def normalize_product_id(value):
    s = str(value).strip().upper()
    m = __import__("re").search(r"(?:D|DEPARTMENT)\s*0*(\d+)", s)
    if m: return f"D{int(m.group(1)):03d}"
    if s.isdigit(): return f"D{int(s):03d}"
    return s


def normalize_store_id(value):
    s = str(value).strip().upper()
    m = __import__("re").search(r"(?:S|STORE)\s*0*(\d+)", s)
    if m: return f"S{int(m.group(1)):03d}"
    if s.isdigit(): return f"S{int(s):03d}"
    return s


def load_history(db_path="retailiq.db", product_id=None, store_id=None):
    import sqlite3
    product_id = normalize_product_id(product_id)
    store_id = normalize_store_id(store_id)
    conn = sqlite3.connect(db_path)
    q = """
        SELECT Date AS week_start,
               product_id,
               store_id,
               Weekly_Sales AS quantity,
               avg_price,
               promotion_flag AS promotion_rate
        FROM fact_transactions
        WHERE product_id = ? AND store_id = ?
        ORDER BY Date
    """
    df = pd.read_sql_query(q, conn, params=[product_id, store_id])
    conn.close()
    if df.empty:
        raise ValueError(f"No historical data found for {product_id} at {store_id}.")
    return df


def forecast_next_weeks(history, product_id, store_id, horizon=4, model=None):
    model = model or load_model()
    product_id = normalize_product_id(product_id)
    store_id = normalize_store_id(store_id)
    h = history.copy()
    h["week_start"] = pd.to_datetime(h["week_start"])
    h = h.sort_values("week_start").reset_index(drop=True)
    if len(h) < 8:
        raise ValueError("At least 8 historical weeks are required.")

    price = float(h["avg_price"].tail(4).mean())
    promo = float(h["promotion_rate"].tail(4).mean())
    working = h[["week_start","quantity","avg_price","promotion_rate"]].copy()
    last = working["week_start"].max()
    out = []

    for step in range(1, int(horizon) + 1):
        d = last + pd.Timedelta(weeks=step)
        q = working["quantity"].astype(float).tolist()
        row = pd.DataFrame([{
            "lag_1": q[-1], "lag_2": q[-2],
            "lag_4": q[-4], "lag_8": q[-8],
            "roll_4": np.mean(q[-4:]),
            "roll_8": np.mean(q[-8:]),
            "avg_price": price,
            "promotion_rate": promo,
            "weekofyear": int(d.isocalendar().week),
            "month": int(d.month)
        }])[FEATURES]
        y = max(0.0, float(model.predict(row)[0]))
        out.append({
            "week_start": d, "product_id": product_id,
            "store_id": store_id, "forecast_quantity": y
        })
        working.loc[len(working)] = [d, y, price, promo]

    return pd.DataFrame(out)


def forecast(product_id, store_id, horizon=4, db_path="retailiq.db"):
    history = load_history(db_path, product_id, store_id)
    return forecast_next_weeks(history, product_id, store_id, horizon)


def load_model_metrics():
    return pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None


if __name__ == "__main__":
    print(forecast("D001", "S001", 4).to_string(index=False))
