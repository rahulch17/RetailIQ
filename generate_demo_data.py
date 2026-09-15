from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DATA = Path("data")
DATA.mkdir(exist_ok=True)

dates = pd.date_range("2024-01-01", "2026-06-30", freq="D")
n_products, n_stores = 60, 10

products = pd.DataFrame({
    "product_id": [f"P{i:03d}" for i in range(1, n_products + 1)],
    "product_name": [f"Product {i:03d}" for i in range(1, n_products + 1)],
    "category": rng.choice(
        ["Beverages", "Snacks", "Grocery", "Personal Care", "Household"],
        n_products
    )
})

stores = pd.DataFrame({
    "store_id": [f"S{i:02d}" for i in range(1, n_stores + 1)],
    "store_name": [f"Store {i:02d}" for i in range(1, n_stores + 1)],
    "city": rng.choice(["Delhi", "Noida", "Gurugram", "Jaipur", "Lucknow"], n_stores),
    "region": rng.choice(["North", "West", "Central"], n_stores)
})

rows = []
for d in dates:
    dow = d.dayofweek
    season = 1 + 0.18 * np.sin(2 * np.pi * d.dayofyear / 365.25)
    for p in products.itertuples():
        for s in stores.itertuples():
            promo = int(rng.random() < (0.20 if dow >= 4 else 0.10))
            base = rng.uniform(8, 45)
            weekend = 1.12 if dow >= 5 else 1.0
            lam = max(1, base * season * weekend * (1.35 if promo else 1.0))
            qty = rng.poisson(lam)
            price = round(rng.uniform(20, 500) * (0.90 if promo else 1.0), 2)
            rows.append((d, p.product_id, s.store_id, qty, price, promo))

transactions = pd.DataFrame(
    rows,
    columns=["date", "product_id", "store_id", "quantity", "price", "promotion_flag"]
)
transactions.to_csv(DATA / "transactions.csv", index=False)
products.to_csv(DATA / "products.csv", index=False)
stores.to_csv(DATA / "stores.csv", index=False)

inventory = (
    transactions.groupby(["date", "product_id", "store_id"], as_index=False)["quantity"]
    .sum()
    .rename(columns={"quantity": "demand"})
)
inventory["inventory"] = (
    inventory["demand"] * rng.uniform(2, 8, len(inventory))
).clip(lower=0).round().astype(int)
inventory[["date", "product_id", "store_id", "inventory"]].to_csv(
    DATA / "inventory.csv", index=False
)

transactions[["date", "product_id", "store_id", "promotion_flag"]].drop_duplicates().to_csv(
    DATA / "promotions.csv", index=False
)

KB = Path("knowledge_base")
KB.mkdir(exist_ok=True)
(KB / "inventory_policy.txt").write_text(
    "Inventory Policy\n"
    "Stores should review inventory cover weekly. Items with low projected cover "
    "should be escalated for replenishment.\n",
    encoding="utf-8"
)
(KB / "markdown_rules.txt").write_text(
    "Markdown Rules\n"
    "Markdown decisions should consider remaining inventory, expected demand, "
    "product age and promotional activity.\n",
    encoding="utf-8"
)
(KB / "returns_policy.txt").write_text(
    "Returns Policy\n"
    "Returned units are recorded as negative transaction quantities and should "
    "be reconciled before demand modelling.\n",
    encoding="utf-8"
)

print("Demo data created in ./data and ./knowledge_base")
