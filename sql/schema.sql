-- RetailIQ star schema
CREATE TABLE IF NOT EXISTS dim_product (
    product_id TEXT PRIMARY KEY,
    Dept INTEGER,
    product_name TEXT,
    category TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS dim_store (
    store_id TEXT PRIMARY KEY,
    Store INTEGER,
    store_name TEXT,
    store_type TEXT,
    store_size REAL
);

CREATE TABLE IF NOT EXISTS dim_inventory (
    store_id TEXT,
    product_id TEXT,
    inventory_units REAL,
    inventory_source TEXT
);

CREATE TABLE IF NOT EXISTS dim_calendar (
    date_id INTEGER PRIMARY KEY,
    Date TIMESTAMP,
    weekofyear INTEGER,
    month INTEGER,
    year INTEGER,
    quarter INTEGER
);

CREATE TABLE IF NOT EXISTS fact_transactions (
    Date TIMESTAMP,
    Store INTEGER,
    Dept INTEGER,
    product_id TEXT,
    store_id TEXT,
    Weekly_Sales REAL,
    IsHoliday INTEGER,
    Temperature REAL,
    Fuel_Price REAL,
    MarkDown1 REAL,
    MarkDown2 REAL,
    MarkDown3 REAL,
    MarkDown4 REAL,
    MarkDown5 REAL,
    CPI REAL,
    Unemployment REAL,
    promotion_flag INTEGER,
    markdown_value REAL,
    avg_price REAL,
    week_start TIMESTAMP,
    quantity REAL,
    revenue REAL,
    promotion_rate REAL
);
