from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

DATA_DIR = Path(r"D:\7th sem\bepractical\Retail\walmart_dataset")
TRAIN_PATH = DATA_DIR / "train.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"   # optional
DB_PATH = Path("retailiq.db")


def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_sources():
    for p in [TRAIN_PATH, FEATURES_PATH, STORES_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Put the Walmart CSV files in the walmart/ folder.")

    train = pd.read_csv(TRAIN_PATH)
    features = pd.read_csv(FEATURES_PATH)
    stores = pd.read_csv(STORES_PATH)

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    train = clean_numeric(train, ["Store", "Dept", "Weekly_Sales"])
    features = clean_numeric(features, [
        "Store", "Temperature", "Fuel_Price", "MarkDown1",
        "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment"
    ])

    stores = clean_numeric(stores, ["Store", "Size"])

    products = None
    if PRODUCTS_PATH.exists():
        products = pd.read_csv(PRODUCTS_PATH)

    return train, features, stores, products


def build_database():
    train, features, stores, products = load_sources()

    train = train.dropna(subset=["Store", "Dept", "Date", "Weekly_Sales"])
    train = train.drop_duplicates(subset=["Store", "Dept", "Date"])

    # Walmart has no item price, inventory table, or true promotion flag.
    # MarkDown1-5 are treated as a promotion/markdown proxy.
    markdown_cols = [c for c in [
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"
    ] if c in features.columns]

    for c in markdown_cols:
        features[c] = features[c].fillna(0)

    features["markdown_value"] = features[markdown_cols].sum(axis=1) if markdown_cols else 0
    features["promotion_flag"] = (features["markdown_value"] > 0).astype(int)

    merged = train.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left",
        suffixes=("", "_feature")
    )

    merged["promotion_flag"] = merged["promotion_flag"].fillna(0).astype(int)
    merged["markdown_value"] = merged["markdown_value"].fillna(0)
    merged["avg_price"] = 1.0  # neutral placeholder: Walmart source has no selling price

    merged["product_id"] = merged["Dept"].astype(int).map(lambda x: f"D{x:03d}")
    merged["store_id"] = merged["Store"].astype(int).map(lambda x: f"S{x:03d}")

    # Dimensions
    dim_store = stores.copy()
    dim_store["store_id"] = dim_store["Store"].astype(int).map(lambda x: f"S{x:03d}")
    dim_store["store_name"] = dim_store["Store"].astype(int).map(lambda x: f"Store {x}")
    dim_store = dim_store.rename(columns={"Type": "store_type", "Size": "store_size"})
    dim_store = dim_store[["store_id", "Store", "store_name", "store_type", "store_size"]]

    departments = sorted(train["Dept"].dropna().astype(int).unique())
    dim_product = pd.DataFrame({
        "product_id": [f"D{x:03d}" for x in departments],
        "Dept": departments,
        "product_name": [f"Department {x}" for x in departments],
        "category": [f"Department {x}" for x in departments],
        "description": [f"Department {x} merchandise" for x in departments],
    })

    if products is not None and {"Dept", "description"}.issubset(products.columns):
        p = products.copy()
        p["Dept"] = pd.to_numeric(p["Dept"], errors="coerce")
        desc = p.groupby("Dept", as_index=False)["description"].first()
        dim_product = dim_product.drop(columns=["description"]).merge(desc, on="Dept", how="left")
        dim_product["description"] = dim_product["description"].fillna(
            dim_product["product_name"] + " merchandise"
        )

    # Optional inventory source. If unavailable, create an explicit compatibility table.
    dim_inventory = pd.DataFrame({
        "store_id": dim_store["store_id"],
        "product_id": [dim_product["product_id"].iloc[0]] * len(dim_store),
        "inventory_units": [np.nan] * len(dim_store),
        "inventory_source": ["not supplied by Walmart dataset"] * len(dim_store)
    })

    fact = merged[[
        "Date", "Store", "Dept", "product_id", "store_id",
        "Weekly_Sales", "IsHoliday", "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment", "promotion_flag", "markdown_value", "avg_price"
    ]].copy()

    fact["week_start"] = fact["Date"]
    fact["quantity"] = fact["Weekly_Sales"]
    fact["revenue"] = fact["Weekly_Sales"]
    fact["promotion_rate"] = fact["promotion_flag"]

    conn = sqlite3.connect(DB_PATH)

    dim_product.to_sql("dim_product", conn, if_exists="replace", index=False)
    dim_store.to_sql("dim_store", conn, if_exists="replace", index=False)
    dim_inventory.to_sql("dim_inventory", conn, if_exists="replace", index=False)
    fact.to_sql("fact_transactions", conn, if_exists="replace", index=False)

    # Calendar dimension
    dates = pd.DataFrame({"Date": sorted(fact["Date"].dropna().unique())})
    dates["date_id"] = pd.to_datetime(dates["Date"]).dt.strftime("%Y%m%d").astype(int)
    dates["weekofyear"] = pd.to_datetime(dates["Date"]).dt.isocalendar().week.astype(int)
    dates["month"] = pd.to_datetime(dates["Date"]).dt.month
    dates["year"] = pd.to_datetime(dates["Date"]).dt.year
    dates["quarter"] = pd.to_datetime(dates["Date"]).dt.quarter
    dim_calendar = dates[["date_id","Date","weekofyear","month","year","quarter"]]
    dim_calendar.to_sql("dim_calendar", conn, if_exists="replace", index=False)

    # Indexes
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_fact_store ON fact_transactions(store_id);
        CREATE INDEX IF NOT EXISTS idx_fact_product ON fact_transactions(product_id);
        CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_transactions(Date);
        CREATE INDEX IF NOT EXISTS idx_fact_store_product_date
            ON fact_transactions(store_id, product_id, Date);
    """)
    conn.commit()
    conn.close()

    print("Database built:", DB_PATH)
    print("Rows:", len(fact))
    print("Stores:", fact["Store"].nunique())
    print("Departments:", fact["Dept"].nunique())
    print("Date range:", fact["Date"].min().date(), "to", fact["Date"].max().date())
    print("Promotion-proxy rows:", int(fact["promotion_flag"].sum()))


if __name__ == "__main__":
    build_database()
