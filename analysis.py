import sqlite3
import pandas as pd
import numpy as np

DB="retailiq.db"

def main():
    conn=sqlite3.connect(DB)
    fact=pd.read_sql_query("SELECT * FROM fact_transactions",conn)
    conn.close()
    fact["Date"]=pd.to_datetime(fact["Date"])
    fact["month"]=fact["Date"].dt.month
    fact["year"]=fact["Date"].dt.year

    print("=== COVERAGE ===")
    print(f"Rows: {len(fact):,}")
    print(f"Stores: {fact.Store.nunique()}")
    print(f"Departments: {fact.Dept.nunique()}")
    print(f"Date range: {fact.Date.min().date()} to {fact.Date.max().date()}")

    print("\n=== SEASONALITY ===")
    season=fact.groupby("month")["Weekly_Sales"].mean().sort_values(ascending=False)
    print(season)

    print("\n=== PROMOTIONAL/MARKDOWN LIFT ===")
    lift=fact.groupby("promotion_flag")["Weekly_Sales"].mean()
    print(lift)
    if 0 in lift.index and 1 in lift.index:
        print(f"Lift: {(lift[1]/lift[0]-1)*100:.2f}%")

    print("\n=== TOP STORES ===")
    print(fact.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False).head(10))

    print("\n=== TOP DEPARTMENTS ===")
    print(fact.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False).head(10))

    print("\nNOTE: Price sensitivity cannot be estimated from Walmart source because no selling-price field exists.")
    print("NOTE: Inventory cover cannot be computed without an inventory-level source.")

if __name__=="__main__":
    main()
