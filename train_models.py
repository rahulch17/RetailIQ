import sqlite3
import pandas as pd
from forecast import train_tree_model

DB_PATH = "retailiq.db"


def load_weekly():
    conn = sqlite3.connect(DB_PATH)
    q = """
        SELECT Date AS week_start,
               product_id, store_id,
               Weekly_Sales AS quantity,
               avg_price,
               promotion_flag AS promotion_rate
        FROM fact_transactions
        ORDER BY product_id, store_id, Date
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df


if __name__ == "__main__":
    weekly = load_weekly()
    model, mape, rmse = train_tree_model(weekly)
    print(f"Tree model: MAPE={mape:.4f}, RMSE={rmse:.4f}")

    try:
        from sequence_model import train_sequence_model
        train_sequence_model(weekly)
        print("LSTM sequence model trained.")
    except Exception as e:
        print("LSTM training skipped:", type(e).__name__, str(e))
        print("The tree model is still ready for the application.")
