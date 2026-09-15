from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_4", "roll_8", "avg_price",
    "promotion_rate", "weekofyear", "month"
]

def make_features(weekly):
    df = weekly.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["product_id", "store_id", "week_start"])

    g = df.groupby(["product_id", "store_id"], group_keys=False)
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = g["quantity"].shift(lag)

    df["roll_4"] = g["quantity"].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )
    df["roll_8"] = g["quantity"].transform(
        lambda s: s.shift(1).rolling(8).mean()
    )
    df["weekofyear"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month

    return df.dropna(subset=FEATURES + ["quantity"])

def forward_split(df, test_fraction=0.2):
    cutoff = df["week_start"].quantile(1 - test_fraction)
    return df[df.week_start < cutoff], df[df.week_start >= cutoff]

def train_tree_model(weekly):
    df = make_features(weekly)
    train, test = forward_split(df)

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[FEATURES], train["quantity"])

    pred = np.maximum(model.predict(test[FEATURES]), 0)
    actual = test["quantity"].clip(lower=1)

    mape = mean_absolute_percentage_error(actual, pred)
    rmse = mean_squared_error(test["quantity"], pred) ** 0.5

    joblib.dump(model, ART / "tree_model.joblib")
    pd.DataFrame([{
        "model": "Tree-based",
        "MAPE": mape,
        "RMSE": rmse
    }]).to_csv(ART / "metrics.csv", index=False)

    return model, mape, rmse

def forecast_next_weeks(history, product_id, store_id, horizon=4):
    model = joblib.load(ART / "tree_model.joblib")

    x = history[
        (history.product_id == product_id) &
        (history.store_id == store_id)
    ].copy()

    x["week_start"] = pd.to_datetime(x["week_start"])
    x = x.sort_values("week_start")

    if len(x) < 8:
        raise ValueError("At least 8 historical weeks are needed.")

    values = list(x["quantity"].astype(float).tail(8))
    prices = list(x["avg_price"].astype(float).tail(8))
    promo = list(x["promotion_rate"].astype(float).tail(8))
    next_week = x["week_start"].max() + pd.Timedelta(weeks=1)

    out = []

    for i in range(horizon):
        date = next_week + pd.Timedelta(weeks=i)

        feats = pd.DataFrame([{
            "lag_1": values[-1],
            "lag_2": values[-2],
            "lag_4": values[-4],
            "lag_8": values[-8],
            "roll_4": np.mean(values[-4:]),
            "roll_8": np.mean(values[-8:]),
            "avg_price": np.mean(prices[-4:]),
            "promotion_rate": np.mean(promo[-4:]),
            "weekofyear": int(date.isocalendar().week),
            "month": date.month,
        }])

        yhat = max(0.0, float(model.predict(feats[FEATURES])[0]))
        out.append({
            "week_start": date,
            "predicted_demand": yhat
        })

        values.append(yhat)
        prices.append(prices[-1])
        promo.append(promo[-1])

    return pd.DataFrame(out)
