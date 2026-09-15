# ============================================================
# RetailIQ - Walmart AI Business Agent
# ============================================================
#
# RetailIQ AI Agent
#
# Tools:
#   1. SQL Analytics
#   2. Demand Forecasting
#   3. Document Retrieval
#
# Dataset:
#   Walmart Store Sales Forecasting
#
# Database:
#   retailiq.db
#
# ============================================================

from pathlib import Path
import os
import re
import json
import sqlite3

import pandas as pd

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

DB_PATH = Path("retailiq.db")

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()


# ============================================================
# GEMINI
# ============================================================

try:

    from google import genai

except ImportError:

    genai = None


# ============================================================
# RETRIEVAL
# ============================================================

try:

    from retrieval import Retriever

except ImportError:

    Retriever = None


# ============================================================
# FORECAST
# ============================================================

try:

    from forecast import forecast_next_weeks

except ImportError:

    forecast_next_weeks = None


# ============================================================
# SQL TOOL
# ============================================================

class SQLTool:
    """
    SQL analytics tool for Walmart data.

    Tables:

        dim_product
        dim_store
        fact_transactions
        fact_weekly_sales
    """

    def __init__(self):

        self.db_path = DB_PATH

    # --------------------------------------------------------
    # Check database
    # --------------------------------------------------------

    def check_database(self):

        if not self.db_path.exists():

            raise FileNotFoundError(
                "retailiq.db was not found.\n\n"
                "Please run:\n"
                "python pipeline.py"
            )

    # --------------------------------------------------------
    # Execute SQL
    # --------------------------------------------------------

    def execute(self, sql):

        self.check_database()

        with sqlite3.connect(
            self.db_path
        ) as connection:

            return pd.read_sql_query(
                sql,
                connection
            )

    # --------------------------------------------------------
    # Products / Departments
    # --------------------------------------------------------

    def get_products(self):

        sql = """
        SELECT
            product_id,
            product_name,
            category,
            Dept
        FROM dim_product
        ORDER BY Dept
        """

        return self.execute(sql)

    # --------------------------------------------------------
    # Stores
    # --------------------------------------------------------

    def get_stores(self):

        sql = """
        SELECT
            store_id,
            store_name,
            Store,
            store_type,
            store_size
        FROM dim_store
        ORDER BY Store
        """

        return self.execute(sql)

    # --------------------------------------------------------
    # Resolve Department
    # --------------------------------------------------------

    def resolve_product(
        self,
        product_text
    ):

        if not product_text:

            return None

        products = self.get_products()

        search = str(
            product_text
        ).strip().lower()

        # Exact ID

        for _, row in products.iterrows():

            if (
                str(row["product_id"]).lower()
                == search
            ):

                return row["product_id"]

        # Exact name

        for _, row in products.iterrows():

            if (
                str(row["product_name"]).lower()
                == search
            ):

                return row["product_id"]

        # Department number

        match = re.search(
            r"\d+",
            search
        )

        if match:

            dept_number = int(
                match.group(0)
            )

            for _, row in products.iterrows():

                try:

                    if int(row["Dept"]) == dept_number:

                        return row["product_id"]

                except Exception:

                    pass

        # Partial name

        for _, row in products.iterrows():

            name = str(
                row["product_name"]
            ).lower()

            if search in name:

                return row["product_id"]

        return None

    # --------------------------------------------------------
    # Resolve Store
    # --------------------------------------------------------

    def resolve_store(
        self,
        store_text
    ):

        if not store_text:

            return None

        stores = self.get_stores()

        search = str(
            store_text
        ).strip().lower()

        # Exact ID

        for _, row in stores.iterrows():

            if (
                str(row["store_id"]).lower()
                == search
            ):

                return row["store_id"]

        # Store number

        match = re.search(
            r"\d+",
            search
        )

        if match:

            store_number = int(
                match.group(0)
            )

            for _, row in stores.iterrows():

                try:

                    if int(row["Store"]) == store_number:

                        return row["store_id"]

                except Exception:

                    pass

        return None

    # ========================================================
    # RUN SQL INTENT
    # ========================================================

    def run_intent(
        self,
        intent,
        top_n=10
    ):

        intent = str(
            intent
        ).lower().strip()

        # ----------------------------------------------------
        # Total sales
        # ----------------------------------------------------

        if intent == "total_sales":

            sql = """
            SELECT
                ROUND(
                    SUM(quantity),
                    2
                ) AS total_sales
            FROM fact_weekly_sales
            """

        # ----------------------------------------------------
        # Total revenue
        # ----------------------------------------------------

        elif intent == "total_revenue":

            sql = """
            SELECT
                ROUND(
                    SUM(revenue),
                    2
                ) AS total_revenue,

                ROUND(
                    SUM(quantity),
                    2
                ) AS total_sales

            FROM fact_weekly_sales
            """

        # ----------------------------------------------------
        # Top departments
        # ----------------------------------------------------

        elif intent == "top_products":

            sql = f"""
            SELECT
                p.product_id,
                p.product_name,
                p.Dept,

                ROUND(
                    SUM(f.quantity),
                    2
                ) AS sales

            FROM fact_weekly_sales f

            LEFT JOIN dim_product p
                ON f.product_id = p.product_id

            GROUP BY
                p.product_id,
                p.product_name,
                p.Dept

            ORDER BY
                sales DESC

            LIMIT {int(top_n)}
            """

        # ----------------------------------------------------
        # Store sales
        # ----------------------------------------------------

        elif intent == "store_sales":

            sql = """
            SELECT
                s.store_id,
                s.store_name,
                s.Store,

                ROUND(
                    SUM(f.quantity),
                    2
                ) AS sales

            FROM fact_weekly_sales f

            LEFT JOIN dim_store s
                ON f.store_id = s.store_id

            GROUP BY
                s.store_id,
                s.store_name,
                s.Store

            ORDER BY
                sales DESC
            """

        # ----------------------------------------------------
        # Department sales
        # ----------------------------------------------------

        elif intent == "department_sales":

            sql = """
            SELECT
                p.product_id,
                p.product_name,
                p.Dept,

                ROUND(
                    SUM(f.quantity),
                    2
                ) AS sales

            FROM fact_weekly_sales f

            LEFT JOIN dim_product p
                ON f.product_id = p.product_id

            GROUP BY
                p.product_id,
                p.product_name,
                p.Dept

            ORDER BY
                sales DESC
            """

        # ----------------------------------------------------
        # Monthly sales
        # ----------------------------------------------------

        elif intent == "monthly_sales":

            sql = """
            SELECT
                year,
                month,

                ROUND(
                    SUM(quantity),
                    2
                ) AS sales

            FROM fact_weekly_sales

            GROUP BY
                year,
                month

            ORDER BY
                year,
                month
            """

        # ----------------------------------------------------
        # Holiday sales
        # ----------------------------------------------------

        elif intent == "holiday_sales":

            sql = """
            SELECT

                CASE
                    WHEN is_holiday = 1
                    THEN 'Holiday'
                    ELSE 'Non-Holiday'
                END AS period,

                ROUND(
                    SUM(quantity),
                    2
                ) AS sales,

                COUNT(*) AS records

            FROM fact_weekly_sales

            GROUP BY
                is_holiday

            ORDER BY
                is_holiday
            """

        # ----------------------------------------------------
        # Promotion / markdown
        # ----------------------------------------------------

        elif intent == "promotion_sales":

            sql = """
            SELECT

                CASE
                    WHEN promotion_rate > 0
                    THEN 'Markdown'
                    ELSE 'No Markdown'
                END AS promotion_status,

                ROUND(
                    AVG(quantity),
                    2
                ) AS average_weekly_sales,

                COUNT(*) AS weeks

            FROM fact_weekly_sales

            GROUP BY
                promotion_status

            ORDER BY
                promotion_status
            """

        # ----------------------------------------------------
        # Store performance
        # ----------------------------------------------------

        elif intent == "store_performance":

            sql = """
            SELECT

                s.store_id,
                s.store_name,
                s.store_type,
                s.store_size,

                ROUND(
                    SUM(f.quantity),
                    2
                ) AS total_sales,

                ROUND(
                    AVG(f.quantity),
                    2
                ) AS average_weekly_sales

            FROM fact_weekly_sales f

            LEFT JOIN dim_store s
                ON f.store_id = s.store_id

            GROUP BY
                s.store_id,
                s.store_name,
                s.store_type,
                s.store_size

            ORDER BY
                total_sales DESC
            """

        # ----------------------------------------------------
        # Department performance
        # ----------------------------------------------------

        elif intent == "department_performance":

            sql = """
            SELECT

                p.product_id,
                p.product_name,
                p.category,

                ROUND(
                    SUM(f.quantity),
                    2
                ) AS total_sales,

                ROUND(
                    AVG(f.quantity),
                    2
                ) AS average_weekly_sales

            FROM fact_weekly_sales f

            LEFT JOIN dim_product p
                ON f.product_id = p.product_id

            GROUP BY
                p.product_id,
                p.product_name,
                p.category

            ORDER BY
                total_sales DESC
            """

        # ----------------------------------------------------
        # Markdown
        # ----------------------------------------------------

        elif intent == "markdown":

            sql = """
            SELECT

                ROUND(
                    AVG(markdown_value),
                    2
                ) AS average_markdown,

                ROUND(
                    MAX(markdown_value),
                    2
                ) AS maximum_markdown

            FROM fact_weekly_sales
            """

        # ----------------------------------------------------
        # Data summary
        # ----------------------------------------------------

        elif intent == "data_summary":

            sql = """
            SELECT

                COUNT(*) AS weekly_records,

                COUNT(
                    DISTINCT product_id
                ) AS departments,

                COUNT(
                    DISTINCT store_id
                ) AS stores,

                MIN(week_start)
                    AS first_week,

                MAX(week_start)
                    AS last_week,

                ROUND(
                    SUM(quantity),
                    2
                ) AS total_sales

            FROM fact_weekly_sales
            """

        # ----------------------------------------------------
        # Default
        # ----------------------------------------------------

        else:

            sql = """
            SELECT

                ROUND(
                    SUM(revenue),
                    2
                ) AS total_revenue,

                ROUND(
                    SUM(quantity),
                    2
                ) AS total_sales

            FROM fact_weekly_sales
            """

        result = self.execute(
            sql
        )

        return result, sql


# ============================================================
# FORECAST TOOL
# ============================================================

class ForecastTool:

    def __init__(self):

        self.db_path = DB_PATH

    def run(
        self,
        product_id,
        store_id,
        horizon=4
    ):

        if not self.db_path.exists():

            raise FileNotFoundError(
                "retailiq.db not found. "
                "Run pipeline.py first."
            )

        if forecast_next_weeks is None:

            raise ImportError(
                "forecast.py could not be imported."
            )

        with sqlite3.connect(
            self.db_path
        ) as connection:

            weekly = pd.read_sql_query(
                """
                SELECT *
                FROM fact_weekly_sales
                """,
                connection
            )

        # Compatibility

        if (
            "promotion_rate"
            not in weekly.columns
        ):

            if "promotion_flag" in weekly.columns:

                weekly["promotion_rate"] = (
                    weekly["promotion_flag"]
                )

            else:

                weekly["promotion_rate"] = 0

        return forecast_next_weeks(
            weekly,
            product_id,
            store_id,
            horizon
        )


# ============================================================
# GEMINI PLANNER
# ============================================================

class GeminiPlanner:

    def __init__(self):

        self.api_key = GEMINI_API_KEY

        self.model_name = GEMINI_MODEL

        self.enabled = (
            bool(self.api_key)
            and genai is not None
        )

        self.client = None

        if self.enabled:

            try:

                self.client = genai.Client(
                    api_key=self.api_key
                )

            except Exception as error:

                print(
                    "Gemini initialization error:",
                    error
                )

                self.enabled = False

    # --------------------------------------------------------
    # Plan
    # --------------------------------------------------------

    def plan(
        self,
        question,
        products=None,
        stores=None
    ):

        if not self.enabled:

            return None

        product_context = ""

        if products is not None:

            for _, row in products.head(100).iterrows():

                product_context += (
                    f"{row['product_id']} = "
                    f"{row['product_name']} "
                    f"(Department {row['Dept']})\n"
                )

        store_context = ""

        if stores is not None:

            for _, row in stores.head(100).iterrows():

                store_context += (
                    f"{row['store_id']} = "
                    f"{row['store_name']} "
                    f"(Store {row['Store']})\n"
                )

        prompt = f"""
You are the planning component of RetailIQ.

You have exactly three tools.

TOOL 1: sql
Use for historical business analytics.

TOOL 2: forecast
Use for future demand or sales prediction.

TOOL 3: retrieval
Use for internal company documents,
policies, SOPs, inventory rules,
return policies, supplier terms,
and markdown rules.

WALMART DATASET:

Store = store
Dept = department/product group
Weekly_Sales = weekly sales
IsHoliday = holiday indicator
MarkDown1-5 = markdown information
Temperature = temperature
Fuel_Price = fuel price
CPI = CPI
Unemployment = unemployment

IMPORTANT:

Return ONLY valid JSON.

Use this format:

{{
    "tool": "sql",
    "reason": "short explanation",
    "intent": "total_revenue",
    "product_id": null,
    "store_id": null,
    "horizon": 4,
    "top_n": 10
}}

Allowed SQL intents:

total_sales
total_revenue
top_products
store_sales
department_sales
monthly_sales
holiday_sales
promotion_sales
store_performance
department_performance
markdown
data_summary

Available departments:

{product_context}

Available stores:

{store_context}

User question:

{question}
"""

        try:

            response = (
                self.client
                .models
                .generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            )

            text = response.text.strip()

            text = re.sub(
                r"^```json\s*",
                "",
                text,
                flags=re.I
            )

            text = re.sub(
                r"\s*```$",
                "",
                text
            )

            plan = json.loads(
                text
            )

            if not isinstance(
                plan,
                dict
            ):

                return None

            if plan.get("tool") not in [
                "sql",
                "forecast",
                "retrieval"
            ]:

                return None

            return plan

        except Exception as error:

            print(
                "Gemini planning error:",
                error
            )

            return None

    # --------------------------------------------------------
    # Retrieval answer
    # --------------------------------------------------------

    def generate_answer(
        self,
        question,
        context
    ):

        if not self.enabled:

            return None

        prompt = f"""
You are RetailIQ's business assistant.

Answer the question using ONLY the
provided internal documentation.

Do not invent information.

Question:

{question}

Internal documentation:

{context}

Give a concise business-friendly answer.
"""

        try:

            response = (
                self.client
                .models
                .generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            )

            return response.text.strip()

        except Exception as error:

            print(
                "Gemini answer error:",
                error
            )

            return None


# ============================================================
# RETAILIQ AGENT
# ============================================================

class RetailIQAgent:

    def __init__(self):

        self.sql_tool = SQLTool()

        self.forecast_tool = ForecastTool()

        self.retriever = (
            Retriever()
            if Retriever is not None
            else None
        )

        self.gemini = GeminiPlanner()

    # ========================================================
    # FALLBACK PLANNER
    # ========================================================

    def fallback_plan(
        self,
        question
    ):

        q = question.lower()

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        retrieval_words = [
            "policy",
            "policies",
            "return",
            "returns",
            "supplier",
            "sop",
            "procedure",
            "inventory policy",
            "markdown rule",
            "rules"
        ]

        if any(
            word in q
            for word in retrieval_words
        ):

            return {
                "tool": "retrieval",
                "reason":
                    "The question asks about an internal policy or procedure."
            }

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        forecast_words = [
            "forecast",
            "forecasting",
            "predict",
            "prediction",
            "future demand",
            "expected demand",
            "next week",
            "next weeks",
            "next month",
            "future sales",
            "will sell"
        ]

        if any(
            word in q
            for word in forecast_words
        ):

            return {
                "tool": "forecast",
                "reason":
                    "The question asks for future sales or demand."
            }

        # ----------------------------------------------------
        # SQL
        # ----------------------------------------------------

        if (
            "top" in q
            and (
                "department" in q
                or "product" in q
            )
        ):

            intent = "top_products"

        elif (
            "highest" in q
            and "store" in q
        ):

            intent = "store_sales"

        elif (
            "best" in q
            and "store" in q
        ):

            intent = "store_sales"

        elif (
            "monthly" in q
            or "month" in q
        ):

            intent = "monthly_sales"

        elif (
            "holiday" in q
            or "holidays" in q
        ):

            intent = "holiday_sales"

        elif (
            "promotion" in q
            or "promotions" in q
            or "markdown" in q
        ):

            intent = "promotion_sales"

        elif (
            "department" in q
            and (
                "sales" in q
                or "performance" in q
            )
        ):

            intent = "department_performance"

        elif (
            "store" in q
            and "sales" in q
        ):

            intent = "store_sales"

        elif (
            "summary" in q
            or "dataset" in q
            or "data coverage" in q
        ):

            intent = "data_summary"

        elif (
            "sales" in q
            or "revenue" in q
        ):

            intent = "total_revenue"

        else:

            intent = "total_revenue"

        return {
            "tool": "sql",
            "reason":
                "The question asks for historical business analytics.",
            "intent": intent
        }

    # ========================================================
    # EXTRACT DEPARTMENT
    # ========================================================

    def extract_product(
        self,
        question,
        products
    ):

        # D001

        match = re.search(
            r"\bD\d+\b",
            question,
            re.I
        )

        if match:

            return (
                self.sql_tool
                .resolve_product(
                    match.group(0)
                )
            )

        # Department 1

        match = re.search(
            r"\bdepartment\s+(\d+)\b",
            question,
            re.I
        )

        if match:

            return (
                self.sql_tool
                .resolve_product(
                    match.group(1)
                )
            )

        # Dept 1

        match = re.search(
            r"\bdept\s+(\d+)\b",
            question,
            re.I
        )

        if match:

            return (
                self.sql_tool
                .resolve_product(
                    match.group(1)
                )
            )

        # Product names

        q = question.lower()

        for _, row in products.iterrows():

            name = str(
                row["product_name"]
            ).lower()

            if name in q:

                return row["product_id"]

        return None

    # ========================================================
    # EXTRACT STORE
    # ========================================================

    def extract_store(
        self,
        question,
        stores
    ):

        # S001

        match = re.search(
            r"\bS\d+\b",
            question,
            re.I
        )

        if match:

            return (
                self.sql_tool
                .resolve_store(
                    match.group(0)
                )
            )

        # Store 1

        match = re.search(
            r"\bstore\s+(\d+)\b",
            question,
            re.I
        )

        if match:

            return (
                self.sql_tool
                .resolve_store(
                    match.group(1)
                )
            )

        return None

    # ========================================================
    # MAIN ANSWER
    # ========================================================

    def answer(
        self,
        question
    ):

        question = str(
            question
        ).strip()

        if not question:

            return {
                "tool": "none",
                "reason": "No question provided.",
                "answer":
                    "Please enter a question.",
                "data": None,
                "sources": []
            }

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        try:

            products = (
                self.sql_tool
                .get_products()
            )

            stores = (
                self.sql_tool
                .get_stores()
            )

        except Exception as error:

            return {
                "tool": "system",
                "reason":
                    "Database could not be loaded.",
                "answer": str(error),
                "data": None,
                "sources": []
            }

        # ----------------------------------------------------
        # Gemini planner
        # ----------------------------------------------------

        plan = self.gemini.plan(
            question,
            products,
            stores
        )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not plan:

            plan = self.fallback_plan(
                question
            )

        tool = str(
            plan.get(
                "tool",
                "sql"
            )
        ).lower().strip()

        reason = plan.get(
            "reason",
            "RetailIQ selected the appropriate business tool."
        )

        # ====================================================
        # RETRIEVAL
        # ====================================================

        if tool == "retrieval":

            if self.retriever is None:

                return {
                    "tool": "retrieval",
                    "reason": reason,
                    "answer":
                        "Retrieval system is not available.",
                    "data": None,
                    "sources": []
                }

            try:

                hits = (
                    self.retriever
                    .search(question)
                )

            except Exception as error:

                return {
                    "tool": "retrieval",
                    "reason": reason,
                    "answer":
                        f"Retrieval failed: {error}",
                    "data": None,
                    "sources": []
                }

            if not hits:

                return {
                    "tool": "retrieval",
                    "reason": reason,
                    "answer":
                        "No matching internal document was found.",
                    "data": None,
                    "sources": []
                }

            context_parts = []

            sources = []

            for hit in hits:

                document = hit.get(
                    "document",
                    "Unknown document"
                )

                text = hit.get(
                    "text",
                    ""
                )

                context_parts.append(
                    f"SOURCE: {document}\n{text}"
                )

                sources.append(
                    document
                )

            context = "\n\n".join(
                context_parts
            )

            answer = (
                self.gemini
                .generate_answer(
                    question,
                    context
                )
            )

            if not answer:

                answer = hits[0].get(
                    "text",
                    "No answer found."
                )

            return {
                "tool": "retrieval",
                "reason": reason,
                "answer": answer,
                "data": None,
                "sources": list(
                    dict.fromkeys(
                        sources
                    )
                )
            }

        # ====================================================
        # FORECAST
        # ====================================================

        if tool == "forecast":

            product_id = plan.get(
                "product_id"
            )

            store_id = plan.get(
                "store_id"
            )

            # ------------------------------------------------
            # Resolve product
            # ------------------------------------------------

            if product_id:

                resolved = (
                    self.sql_tool
                    .resolve_product(
                        product_id
                    )
                )

                if resolved:

                    product_id = resolved

            if not product_id:

                product_id = (
                    self.extract_product(
                        question,
                        products
                    )
                )

            # ------------------------------------------------
            # Resolve store
            # ------------------------------------------------

            if store_id:

                resolved = (
                    self.sql_tool
                    .resolve_store(
                        store_id
                    )
                )

                if resolved:

                    store_id = resolved

            if not store_id:

                store_id = (
                    self.extract_store(
                        question,
                        stores
                    )
                )

            # ------------------------------------------------
            # Product missing
            # ------------------------------------------------

            if not product_id:

                return {
                    "tool": "forecast",
                    "reason": reason,
                    "answer":
                        "Please specify a department.\n\n"
                        "Example:\n"
                        "Forecast Department 1 for Store 1 "
                        "for the next 4 weeks.",
                    "data": None,
                    "sources": []
                }

            # ------------------------------------------------
            # Store missing
            # ------------------------------------------------

            if not store_id:

                return {
                    "tool": "forecast",
                    "reason": reason,
                    "answer":
                        "Please specify a store.\n\n"
                        "Example:\n"
                        "Forecast Department 1 for Store 1 "
                        "for the next 4 weeks.",
                    "data": None,
                    "sources": []
                }

            # ------------------------------------------------
            # Horizon
            # ------------------------------------------------

            try:

                horizon = int(
                    plan.get(
                        "horizon",
                        4
                    )
                )

            except Exception:

                horizon = 4

            horizon = min(
                max(
                    horizon,
                    1
                ),
                12
            )

            # ------------------------------------------------
            # Generate forecast
            # ------------------------------------------------

            try:

                forecast_result = (
                    self.forecast_tool
                    .run(
                        product_id,
                        store_id,
                        horizon
                    )
                )

            except Exception as error:

                return {
                    "tool": "forecast",
                    "reason": reason,
                    "answer":
                        f"Forecast could not be generated: {error}",
                    "data": None,
                    "sources": []
                }

            # ------------------------------------------------
            # Clean forecast DataFrame
            # ------------------------------------------------

            if isinstance(
                forecast_result,
                pd.DataFrame
            ):

                forecast_df = (
                    forecast_result
                    .copy()
                )

                if "week_start" in forecast_df.columns:

                    forecast_df[
                        "week_start"
                    ] = pd.to_datetime(
                        forecast_df[
                            "week_start"
                        ],
                        errors="coerce"
                    ).dt.strftime(
                        "%Y-%m-%d"
                    )

                for column in forecast_df.columns:

                    if (
                        column != "week_start"
                        and pd.api.types.is_numeric_dtype(
                            forecast_df[column]
                        )
                    ):

                        forecast_df[column] = (
                            forecast_df[column]
                            .round(2)
                        )

            else:

                forecast_df = None

            answer = (
                f"Forecast generated successfully.\n\n"
                f"Department: {product_id}\n"
                f"Store: {store_id}\n"
                f"Horizon: {horizon} weeks"
            )

            return {
                "tool": "forecast",
                "reason": reason,
                "answer": answer,
                "data": forecast_df,
                "sources": []
            }

        # ====================================================
        # SQL
        # ====================================================

        if tool == "sql":

            intent = plan.get(
                "intent",
                "total_revenue"
            )

            try:

                top_n = int(
                    plan.get(
                        "top_n",
                        10
                    )
                )

            except Exception:

                top_n = 10

            top_n = min(
                max(
                    top_n,
                    1
                ),
                50
            )

            try:

                result, sql = (
                    self.sql_tool
                    .run_intent(
                        intent,
                        top_n
                    )
                )

            except Exception as error:

                return {
                    "tool": "sql",
                    "reason": reason,
                    "answer":
                        f"SQL execution failed: {error}",
                    "data": None,
                    "sources": []
                }

            # ------------------------------------------------
            # Generate short text answer
            # ------------------------------------------------

            answer = None

            if (
                self.gemini.enabled
                and not result.empty
            ):

                try:

                    result_text = (
                        result
                        .head(20)
                        .to_string(
                            index=False
                        )
                    )

                    prompt = f"""
You are RetailIQ.

Answer this business question
using ONLY the SQL result.

Do not invent values.

Question:
{question}

SQL result:
{result_text}

Give a short, clear business answer.
Do not reproduce the entire table.
"""

                    response = (
                        self.gemini
                        .client
                        .models
                        .generate_content(
                            model=self.gemini.model_name,
                            contents=prompt
                        )
                    )

                    answer = (
                        response.text
                        .strip()
                    )

                except Exception:

                    answer = None

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            if not answer:

                answer = (
                    "SQL analysis completed successfully."
                )

            return {
                "tool": "sql",
                "reason": reason,
                "answer": answer,
                "data": result,
                "sources": []
            }

        # ====================================================
        # UNKNOWN
        # ====================================================

        return {
            "tool": "system",
            "reason":
                "Unknown tool selected.",
            "answer":
                "RetailIQ could not determine the correct tool.",
            "data": None,
            "sources": []
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "RetailIQ Walmart Agent Test"
    )

    print(
        "=" * 70
    )

    agent = RetailIQAgent()

    questions = [

        "Which departments have the highest sales?",

        "Which store has the highest sales?",

        "Show monthly sales.",

        "What are the markdown effects?",

        "Forecast Department 1 for Store 1 for the next 4 weeks.",

        "What is the return policy?"

    ]

    for question in questions:

        print(
            "\nQUESTION:",
            question
        )

        result = agent.answer(
            question
        )

        print(
            "\nTOOL:",
            result.get("tool")
        )

        print(
            "REASON:",
            result.get("reason")
        )

        print(
            "ANSWER:",
            result.get("answer")
        )

        data = result.get(
            "data"
        )

        if isinstance(
            data,
            pd.DataFrame
        ):

            print(
                "\nDATA:"
            )

            print(
                data.head(10)
            )

        if result.get(
            "sources"
        ):

            print(
                "\nSOURCES:",
                result.get("sources")
            )