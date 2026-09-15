from pathlib import Path
import sqlite3
import pandas as pd

DATA = Path("data")
DB = Path("retailiq.db")

def clean_transactions(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["promotion_flag"] = (
        pd.to_numeric(df["promotion_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return df.dropna(subset=["date", "product_id", "store_id"])

def build_weekly_sales(transactions):
    x = transactions.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["week_start"] = x["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    x["revenue"] = x["quantity"] * x["price"]

    return x.groupby(
        ["week_start", "product_id", "store_id"], as_index=False
    ).agg(
        quantity=("quantity", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("price", "mean"),
        promotion_rate=("promotion_flag", "mean"),
    )

def build_db():
    tx = clean_transactions(pd.read_csv(DATA / "transactions.csv"))
    products = pd.read_csv(DATA / "products.csv")
    stores = pd.read_csv(DATA / "stores.csv")
    inventory_path = DATA / "inventory.csv"

    weekly = build_weekly_sales(tx)

    with sqlite3.connect(DB) as con:
        products.to_sql("dim_product", con, if_exists="replace", index=False)
        stores.to_sql("dim_store", con, if_exists="replace", index=False)
        weekly.to_sql("fact_weekly_sales", con, if_exists="replace", index=False)

        if inventory_path.exists():
            inventory = pd.read_csv(inventory_path)
            inventory["date"] = pd.to_datetime(inventory["date"], errors="coerce")
            inventory.to_sql("fact_inventory", con, if_exists="replace", index=False)

        con.executescript("""
        CREATE INDEX IF NOT EXISTS idx_sales_week
        ON fact_weekly_sales(week_start);

        CREATE INDEX IF NOT EXISTS idx_sales_product_store
        ON fact_weekly_sales(product_id, store_id);

        CREATE INDEX IF NOT EXISTS idx_product_category
        ON dim_product(category);

        CREATE INDEX IF NOT EXISTS idx_store_city
        ON dim_store(city);
        """)

    print(f"Built {DB}")

if __name__ == "__main__":
    build_db()
