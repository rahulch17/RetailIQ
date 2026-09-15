# ============================================================
# RetailIQ - Walmart Data Pipeline
# Designed for Walmart Recruiting Store Sales dataset
# ============================================================

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Put the folder containing train.csv, features.csv and
# stores.csv here.
#
# Example:
# DATA_DIR = r"D:\7th sem\bepractical\Retail\walmart"
# ------------------------------------------------------------

DATA_DIR = Path(r"D:\7th sem\bepractical\Retail\walmart_dataset")


TRAIN_PATH = DATA_DIR / "train.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"

# SQLite database used by RetailIQ
DB_PATH = Path("retailiq.db")


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("RetailIQ - Walmart Data Pipeline")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [TRAIN_PATH, FEATURES_PATH, STORES_PATH]:

        if not path.exists():

            raise FileNotFoundError(
                f"\nFile not found:\n{path.absolute()}\n\n"
                "Please check DATA_DIR in pipeline.py."
            )

    print("\nLoading Walmart dataset...")

    train = pd.read_csv(TRAIN_PATH)
    features = pd.read_csv(FEATURES_PATH)
    stores = pd.read_csv(STORES_PATH)

    print(f"\nTrain rows: {len(train):,}")
    print(f"Feature rows: {len(features):,}")
    print(f"Store rows: {len(stores):,}")

    print("\nTrain columns:")
    for column in train.columns:
        print(f"  - {column}")

    return train, features, stores


# ============================================================
# 3. CLEAN DATA
# ============================================================

def clean_data(train, features, stores):

    print("\n" + "=" * 70)
    print("Cleaning Walmart Data")
    print("=" * 70)

    train = train.copy()
    features = features.copy()
    stores = stores.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    train.columns = train.columns.astype(str).str.strip()
    features.columns = features.columns.astype(str).str.strip()
    stores.columns = stores.columns.astype(str).str.strip()

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    train["Date"] = pd.to_datetime(
        train["Date"],
        errors="coerce"
    )

    features["Date"] = pd.to_datetime(
        features["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    train["Store"] = pd.to_numeric(
        train["Store"],
        errors="coerce"
    )

    train["Dept"] = pd.to_numeric(
        train["Dept"],
        errors="coerce"
    )

    train["Weekly_Sales"] = pd.to_numeric(
        train["Weekly_Sales"],
        errors="coerce"
    )

    stores["Store"] = pd.to_numeric(
        stores["Store"],
        errors="coerce"
    )

    stores["Size"] = pd.to_numeric(
        stores["Size"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_train = [
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday"
    ]

    missing = [
        col
        for col in required_train
        if col not in train.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required train columns:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    before = len(train)

    train = train.dropna(
        how="all"
    )

    print(
        f"\nRemoved empty train rows: "
        f"{before - len(train):,}"
    )

    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    before = len(train)

    train = train.drop_duplicates(
        subset=[
            "Store",
            "Dept",
            "Date"
        ]
    )

    print(
        f"Removed duplicate observations: "
        f"{before - len(train):,}"
    )

    # --------------------------------------------------------
    # Remove missing essential values
    # --------------------------------------------------------

    before = len(train)

    train = train.dropna(
        subset=[
            "Store",
            "Dept",
            "Date",
            "Weekly_Sales"
        ]
    )

    print(
        f"Removed incomplete rows: "
        f"{before - len(train):,}"
    )

    # --------------------------------------------------------
    # Remove impossible sales
    #
    # Walmart can contain negative Weekly_Sales in some
    # contexts, but for demand forecasting we keep the
    # original observations rather than silently changing
    # them.
    # --------------------------------------------------------

    print(
        "\nNegative sales observations:",
        int((train["Weekly_Sales"] < 0).sum())
    )

    # --------------------------------------------------------
    # Clean features
    # --------------------------------------------------------

    feature_numeric_columns = [
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment"
    ]

    for column in feature_numeric_columns:

        if column in features.columns:

            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Clean stores
    # --------------------------------------------------------

    if "Type" in stores.columns:

        stores["Type"] = (
            stores["Type"]
            .astype(str)
            .str.strip()
        )

    print(
        f"\nFinal train rows: {len(train):,}"
    )

    print(
        f"Date range: "
        f"{train['Date'].min().date()} "
        f"to "
        f"{train['Date'].max().date()}"
    )

    return train, features, stores


# ============================================================
# 4. CREATE PRODUCT DIMENSION
# ============================================================

def create_product_dimension(train):

    print("\nCreating product dimension...")

    # Walmart uses Dept as the product/category identifier.
    products = (
        train[["Dept"]]
        .drop_duplicates()
        .sort_values("Dept")
        .reset_index(drop=True)
    )

    products["product_id"] = products["Dept"].apply(
        lambda x: f"D{int(x):03d}"
    )

    products["product_name"] = (
        "Department " +
        products["Dept"].astype(int).astype(str)
    )

    # Department becomes the category.
    products["category"] = (
        "Department " +
        products["Dept"].astype(int).astype(str)
    )

    products = products[
        [
            "product_id",
            "product_name",
            "category",
            "Dept"
        ]
    ]

    return products


# ============================================================
# 5. CREATE STORE DIMENSION
# ============================================================

def create_store_dimension(stores):

    print("Creating store dimension...")

    stores = stores.copy()

    stores["store_id"] = stores["Store"].apply(
        lambda x: f"S{int(x):03d}"
    )

    stores["store_name"] = (
        "Store " +
        stores["Store"].astype(int).astype(str)
    )

    stores = stores.rename(
        columns={
            "Type": "store_type",
            "Size": "store_size"
        }
    )

    available_columns = [
        "store_id",
        "store_name",
        "Store",
        "store_type",
        "store_size"
    ]

    stores = stores[
        [
            col
            for col in available_columns
            if col in stores.columns
        ]
    ]

    return stores


# ============================================================
# 6. MERGE TRAIN + FEATURES
# ============================================================

def merge_features(train, features):

    print("\nMerging sales and external features...")

    train = train.copy()
    features = features.copy()

    feature_columns = [
        "Store",
        "Date",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment"
    ]

    feature_columns = [
        col
        for col in feature_columns
        if col in features.columns
    ]

    features_small = features[
        feature_columns
    ].drop_duplicates(
        subset=["Store", "Date"]
    )

    merged = train.merge(
        features_small,
        on=["Store", "Date"],
        how="left"
    )

    print(
        f"Merged rows: {len(merged):,}"
    )

    # --------------------------------------------------------
    # Create markdown promotion proxy
    #
    # Walmart dataset does not have a simple promotion flag.
    # We use markdown information as a promotion/markdown
    # indicator.
    # --------------------------------------------------------

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    available_markdowns = [
        col
        for col in markdown_columns
        if col in merged.columns
    ]

    if available_markdowns:

        merged["promotion_flag"] = (
            merged[available_markdowns]
            .fillna(0)
            .sum(axis=1)
            > 0
        ).astype(int)

    else:

        merged["promotion_flag"] = 0

    # --------------------------------------------------------
    # Total markdown value
    # --------------------------------------------------------

    if available_markdowns:

        merged["markdown_value"] = (
            merged[available_markdowns]
            .fillna(0)
            .sum(axis=1)
        )

    else:

        merged["markdown_value"] = 0.0

    return merged


# ============================================================
# 7. CREATE WEEKLY SALES FACT TABLE
# ============================================================

def create_weekly_sales(
    merged,
    products,
    stores
):

    print("\nCreating weekly sales fact table...")

    x = merged.copy()

    # --------------------------------------------------------
    # Product ID
    # --------------------------------------------------------

    product_map = dict(
        zip(
            products["Dept"],
            products["product_id"]
        )
    )

    x["product_id"] = (
        x["Dept"]
        .map(product_map)
    )

    # --------------------------------------------------------
    # Store ID
    # --------------------------------------------------------

    store_map = dict(
        zip(
            stores["Store"],
            stores["store_id"]
        )
    )

    x["store_id"] = (
        x["Store"]
        .map(store_map)
    )

    # --------------------------------------------------------
    # Walmart training data is already weekly.
    # Use Date as week_start.
    # --------------------------------------------------------

    x["week_start"] = (
        x["Date"]
        .dt.to_period("W-SUN")
        .apply(
            lambda p: p.start_time
        )
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregation = {
        "Weekly_Sales": "sum",
        "promotion_flag": "mean",
        "markdown_value": "mean"
    }

    if "Temperature" in x.columns:
        aggregation["Temperature"] = "mean"

    if "Fuel_Price" in x.columns:
        aggregation["Fuel_Price"] = "mean"

    if "CPI" in x.columns:
        aggregation["CPI"] = "mean"

    if "Unemployment" in x.columns:
        aggregation["Unemployment"] = "mean"

    if "IsHoliday" in x.columns:
        aggregation["IsHoliday"] = "max"

    weekly = (
        x.groupby(
            [
                "week_start",
                "product_id",
                "store_id"
            ],
            as_index=False
        )
        .agg(aggregation)
    )
    weekly["promotion_rate"] = weekly["promotion_flag"]

    # --------------------------------------------------------
    # Rename sales to RetailIQ quantity/revenue
    # --------------------------------------------------------

    weekly = weekly.rename(
        columns={
            "Weekly_Sales": "quantity"
        }
    )

    # Weekly sales are the commercial value.
    weekly["revenue"] = weekly["quantity"]

    # --------------------------------------------------------
    # Walmart does not provide item selling price.
    #
    # We use a neutral constant rather than inventing a price.
    # This keeps compatibility with the existing forecast
    # pipeline.
    # --------------------------------------------------------

    weekly["avg_price"] = 1.0

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    weekly["week_of_year"] = (
        weekly["week_start"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    weekly["month"] = (
        weekly["week_start"]
        .dt.month
    )

    weekly["year"] = (
        weekly["week_start"]
        .dt.year
    )

    weekly["quarter"] = (
        weekly["week_start"]
        .dt.quarter
    )

    weekly["is_holiday"] = (
        weekly["IsHoliday"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    weekly = weekly.sort_values(
        [
            "product_id",
            "store_id",
            "week_start"
        ]
    ).reset_index(drop=True)

    print(
        f"Weekly records created: "
        f"{len(weekly):,}"
    )

    return weekly


# ============================================================
# 8. CREATE TRANSACTION/OBSERVATION FACT TABLE
# ============================================================

def create_transaction_fact(
    merged,
    products,
    stores
):

    print(
        "Creating sales observation fact table..."
    )

    x = merged.copy()

    # --------------------------------------------------------
    # Product mapping
    # --------------------------------------------------------

    product_map = dict(
        zip(
            products["Dept"],
            products["product_id"]
        )
    )

    x["product_id"] = (
        x["Dept"]
        .map(product_map)
    )

    # --------------------------------------------------------
    # Store mapping
    # --------------------------------------------------------

    store_map = dict(
        zip(
            stores["Store"],
            stores["store_id"]
        )
    )

    x["store_id"] = (
        x["Store"]
        .map(store_map)
    )

    # --------------------------------------------------------
    # This dataset is weekly rather than transaction-level.
    # We therefore store each Walmart weekly observation.
    # --------------------------------------------------------

    columns = [
        "Date",
        "Store",
        "Dept",
        "product_id",
        "store_id",
        "Weekly_Sales",
        "IsHoliday",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "promotion_flag",
        "markdown_value"
    ]

    available = [
        col
        for col in columns
        if col in x.columns
    ]

    observations = x[
        available
    ].copy()

    return observations


# ============================================================
# 9. CREATE DATABASE
# ============================================================

def create_database(
    observations,
    products,
    stores,
    weekly
):

    print(
        "\n" + "=" * 70
    )

    print(
        "Creating RetailIQ SQLite database..."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Remove old database
    # --------------------------------------------------------

    if DB_PATH.exists():

        print(
            "\nRemoving old retailiq.db..."
        )

        DB_PATH.unlink()

    # --------------------------------------------------------
    # Create database
    # --------------------------------------------------------

    with sqlite3.connect(DB_PATH) as connection:

        # ----------------------------------------------------
        # Product dimension
        # ----------------------------------------------------

        products.to_sql(
            "dim_product",
            connection,
            if_exists="replace",
            index=False
        )

        # ----------------------------------------------------
        # Store dimension
        # ----------------------------------------------------

        stores.to_sql(
            "dim_store",
            connection,
            if_exists="replace",
            index=False
        )

        # ----------------------------------------------------
        # Observation fact
        # ----------------------------------------------------

        observations.to_sql(
            "fact_transactions",
            connection,
            if_exists="replace",
            index=False
        )

        # ----------------------------------------------------
        # Weekly sales fact
        # ----------------------------------------------------

        weekly.to_sql(
            "fact_weekly_sales",
            connection,
            if_exists="replace",
            index=False
        )

        # ----------------------------------------------------
        # Indexes
        # ----------------------------------------------------

        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS
            idx_transaction_date
            ON fact_transactions(Date);

            CREATE INDEX IF NOT EXISTS
            idx_transaction_product
            ON fact_transactions(product_id);

            CREATE INDEX IF NOT EXISTS
            idx_transaction_store
            ON fact_transactions(store_id);

            CREATE INDEX IF NOT EXISTS
            idx_weekly_date
            ON fact_weekly_sales(week_start);

            CREATE INDEX IF NOT EXISTS
            idx_weekly_product
            ON fact_weekly_sales(product_id);

            CREATE INDEX IF NOT EXISTS
            idx_weekly_store
            ON fact_weekly_sales(store_id);

            CREATE INDEX IF NOT EXISTS
            idx_product_category
            ON dim_product(category);

            CREATE INDEX IF NOT EXISTS
            idx_store_type
            ON dim_store(store_type);
            """
        )

    print(
        f"\nDatabase created successfully:"
    )

    print(
        DB_PATH.absolute()
    )


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

def print_summary(
    merged,
    products,
    stores,
    weekly
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RETAILIQ WALMART DATA SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSales observations: "
        f"{len(merged):,}"
    )

    print(
        f"Departments: "
        f"{products['product_id'].nunique():,}"
    )

    print(
        f"Stores: "
        f"{stores['store_id'].nunique():,}"
    )

    print(
        f"Weekly records: "
        f"{len(weekly):,}"
    )

    print(
        f"Total weekly sales: "
        f"{weekly['quantity'].sum():,.2f}"
    )

    print(
        f"Date start: "
        f"{merged['Date'].min().date()}"
    )

    print(
        f"Date end: "
        f"{merged['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Number of weeks
    # --------------------------------------------------------

    weeks = (
        weekly["week_start"]
        .nunique()
    )

    print(
        f"Number of weeks: "
        f"{weeks:,}"
    )

    # --------------------------------------------------------
    # Series coverage
    # --------------------------------------------------------

    series_counts = (
        weekly
        .groupby(
            [
                "product_id",
                "store_id"
            ]
        )
        .size()
    )

    print(
        "\nProduct-store series:"
    )

    print(
        f"  Total series: "
        f"{len(series_counts):,}"
    )

    print(
        f"  Series with >= 8 weeks: "
        f"{(series_counts >= 8).sum():,}"
    )

    print(
        f"  Series with < 8 weeks: "
        f"{(series_counts < 8).sum():,}"
    )

    # --------------------------------------------------------
    # Promotion information
    # --------------------------------------------------------

    promotion_rows = (
        merged["promotion_flag"]
        .sum()
    )

    print(
        "\nMarkdown/promotion observations:"
    )

    print(
        f"  Rows with markdown activity: "
        f"{int(promotion_rows):,}"
    )

    # --------------------------------------------------------
    # Top departments
    # --------------------------------------------------------

    print(
        "\nTop departments by sales:"
    )

    top_departments = (
        merged
        .groupby("Dept")["Weekly_Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    for dept, sales in top_departments.items():

        print(
            f"  Department {int(dept)}: "
            f"{sales:,.2f}"
        )

    # --------------------------------------------------------
    # Top stores
    # --------------------------------------------------------

    print(
        "\nTop stores by sales:"
    )

    top_stores = (
        merged
        .groupby("Store")["Weekly_Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    for store, sales in top_stores.items():

        print(
            f"  Store {int(store)}: "
            f"{sales:,.2f}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train, features, stores = load_data()

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    train, features, stores = clean_data(
        train,
        features,
        stores
    )

    # --------------------------------------------------------
    # Create dimensions
    # --------------------------------------------------------

    products = create_product_dimension(
        train
    )

    store_dimension = create_store_dimension(
        stores
    )

    # --------------------------------------------------------
    # Merge external features
    # --------------------------------------------------------

    merged = merge_features(
        train,
        features
    )

    # --------------------------------------------------------
    # Create weekly fact
    # --------------------------------------------------------

    weekly = create_weekly_sales(
        merged,
        products,
        store_dimension
    )

    # --------------------------------------------------------
    # Create observation fact
    # --------------------------------------------------------

    observations = create_transaction_fact(
        merged,
        products,
        store_dimension
    )

    # --------------------------------------------------------
    # Create SQLite database
    # --------------------------------------------------------

    create_database(
        observations,
        products,
        store_dimension,
        weekly
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        merged,
        products,
        store_dimension,
        weekly
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()