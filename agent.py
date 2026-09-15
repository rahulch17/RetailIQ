import re
import sqlite3
from pathlib import Path
import pandas as pd

from retrieval import Retriever
from forecast import forecast_next_weeks

DB = Path("retailiq.db")

class SQLTool:
    def run(self, sql):
        with sqlite3.connect(DB) as con:
            return pd.read_sql_query(sql, con)

class ForecastTool:
    def run(self, product_id, store_id, horizon):
        with sqlite3.connect(DB) as con:
            weekly = pd.read_sql(
                "SELECT * FROM fact_weekly_sales",
                con
            )
        return forecast_next_weeks(
            weekly, product_id, store_id, horizon
        )

class RetailIQAgent:
    """
    Planner/executor agent.

    For the MVP, routing is deterministic and auditable.
    A production version can replace choose_tool() with an LLM planner
    while retaining the same SQL/forecast/retrieval tools.
    """

    def __init__(self):
        self.sql = SQLTool()
        self.forecast = ForecastTool()
        self.retriever = Retriever()

    def choose_tool(self, question):
        q = question.lower()

        policy_words = [
            "policy", "rule", "returns", "return", "supplier",
            "markdown", "procedure", "sop"
        ]
        future_words = [
            "forecast", "predict", "next week", "next month",
            "future", "expected demand", "will sell"
        ]

        if any(w in q for w in policy_words):
            return (
                "retrieval",
                "The question asks about an internal policy or procedure."
            )

        if any(w in q for w in future_words):
            return (
                "forecast",
                "The question asks about future demand."
            )

        return (
            "sql",
            "The question asks for historical or analytical business data."
        )

    def answer(self, question):
        tool, reason = self.choose_tool(question)

        if tool == "retrieval":
            hits = self.retriever.search(question)

            if not hits:
                return {
                    "tool": tool,
                    "reason": reason,
                    "answer": "No matching policy document was found.",
                    "sources": [],
                }

            best = hits[0]

            return {
                "tool": tool,
                "reason": reason,
                "answer": best["text"].replace("\n", " "),
                "sources": [best["document"]],
            }

        if tool == "forecast":
            products = re.findall(
                r"P\d{3}",
                question.upper()
            )
            stores = re.findall(
                r"S\d{2}",
                question.upper()
            )

            if not products or not stores:
                return {
                    "tool": tool,
                    "reason": reason,
                    "answer": (
                        "Please include a product ID such as P001 "
                        "and a store ID such as S01."
                    ),
                    "sources": [],
                }

            horizon_match = re.search(
                r"(\d+)\s*week",
                question.lower()
            )
            horizon = int(horizon_match.group(1)) if horizon_match else 4
            horizon = min(horizon, 12)

            result = self.forecast.run(
                products[0], stores[0], horizon
            )

            return {
                "tool": tool,
                "reason": reason,
                "answer": result,
                "sources": ["forecast model"],
            }

        # Safe template-based SQL translator for the MVP.
        q = question.lower()

        if "top" in q and "product" in q:
            n_match = re.search(r"top\s+(\d+)", q)
            n = int(n_match.group(1)) if n_match else 5
            n = min(max(n, 1), 20)

            sql = f"""
            SELECT p.product_id,
                   p.product_name,
                   p.category,
                   ROUND(SUM(f.revenue), 2) AS revenue,
                   SUM(f.quantity) AS units
            FROM fact_weekly_sales f
            JOIN dim_product p
              ON f.product_id = p.product_id
            GROUP BY p.product_id, p.product_name, p.category
            ORDER BY revenue DESC
            LIMIT {n}
            """

        elif "store" in q and (
            "revenue" in q or "sales" in q
        ):
            sql = """
            SELECT s.store_id,
                   s.store_name,
                   s.city,
                   ROUND(SUM(f.revenue), 2) AS revenue,
                   SUM(f.quantity) AS units
            FROM fact_weekly_sales f
            JOIN dim_store s
              ON f.store_id = s.store_id
            GROUP BY s.store_id, s.store_name, s.city
            ORDER BY revenue DESC
            """

        elif "category" in q:
            sql = """
            SELECT p.category,
                   ROUND(SUM(f.revenue), 2) AS revenue,
                   SUM(f.quantity) AS units
            FROM fact_weekly_sales f
            JOIN dim_product p
              ON f.product_id = p.product_id
            GROUP BY p.category
            ORDER BY revenue DESC
            """

        else:
            sql = """
            SELECT ROUND(SUM(revenue), 2) AS revenue,
                   SUM(quantity) AS units
            FROM fact_weekly_sales
            """

        result = self.sql.run(sql)

        return {
            "tool": tool,
            "reason": reason,
            "answer": result,
            "sql": sql,
            "sources": ["star schema"],
        }
