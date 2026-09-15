# ============================================================
# RetailIQ - Final AI Business Assistant
# ============================================================

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from forecast import forecast

try:
    from retrieval import Retriever
except Exception:
    Retriever = None


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DB_PATH = Path("retailiq.db")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

retriever = None

if Retriever is not None:
    try:
        retriever = Retriever()
    except Exception as e:
        print(f"RAG initialization warning: {e}")
        retriever = None


# ============================================================
# GEMINI INITIALIZATION
# ============================================================

gemini_model = None

if GEMINI_API_KEY:

    try:
        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        gemini_model = client.models

    except Exception as e:
        print(
            f"Gemini initialization warning: {e}"
        )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(
        DB_PATH
    )


# ============================================================
# SQL INTENT DETECTION
# ============================================================

def detect_sql_intent(question):

    q = question.lower().strip()

    # --------------------------------------------------------
    # Store ranking
    # --------------------------------------------------------

    if (
        "which store" in q
        or "best store" in q
        or "top store" in q
        or "highest sales store" in q
        or "store has the highest" in q
        or (
            "store" in q
            and (
                "highest sales" in q
                or "most sales" in q
                or "maximum sales" in q
            )
        )
    ):
        return "store_performance"

    # --------------------------------------------------------
    # Department ranking
    # --------------------------------------------------------

    if (
        "which department" in q
        or "best department" in q
        or "top department" in q
        or "highest department" in q
        or (
            "department" in q
            and (
                "highest sales" in q
                or "most sales" in q
                or "maximum sales" in q
            )
        )
    ):
        return "department_performance"

    # --------------------------------------------------------
    # Store sales
    # --------------------------------------------------------

    if (
        "store" in q
        and (
            "sales" in q
            or "revenue" in q
            or "performance" in q
        )
    ):
        return "store_sales"

    # --------------------------------------------------------
    # Department sales
    # --------------------------------------------------------

    if (
        "department" in q
        and (
            "sales" in q
            or "revenue" in q
            or "performance" in q
        )
    ):
        return "department_sales"

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    if (
        "monthly sales" in q
        or "sales by month" in q
        or "month wise sales" in q
        or "monthly revenue" in q
    ):
        return "monthly_sales"

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    if (
        "holiday sales" in q
        or "holiday performance" in q
        or "sales on holidays" in q
        or "holiday" in q and "sales" in q
    ):
        return "holiday_sales"

    # --------------------------------------------------------
    # Promotion / markdown
    # --------------------------------------------------------

    if (
        "promotion" in q
        or "promotions" in q
        or "markdown" in q
        or "mark down" in q
    ):
        return "promotion_sales"

    # --------------------------------------------------------
    # Data summary
    # --------------------------------------------------------

    if (
        "how many rows" in q
        or "how much data" in q
        or "data summary" in q
        or "dataset summary" in q
        or "number of stores" in q
        or "number of departments" in q
        or "date range" in q
    ):
        return "data_summary"

    # --------------------------------------------------------
    # Generic total
    # --------------------------------------------------------

    if (
        "total sales" in q
        or "overall sales" in q
        or "total revenue" in q
        or "overall revenue" in q
    ):
        return "total_sales"

    return None


# ============================================================
# SQL TOOL
# ============================================================

def run_sql(question):

    intent = detect_sql_intent(
        question
    )

    if intent is None:

        return {
            "tool": "SQL",
            "reason": (
                "No matching SQL analytics intent was identified."
            ),
            "answer": (
                "I could not identify the requested business metric."
            ),
            "data": None,
            "sources": []
        }

    conn = get_connection()

    try:

        # ----------------------------------------------------
        # Total sales
        # ----------------------------------------------------

        if intent == "total_sales":

            query = """
                SELECT
                    SUM(Weekly_Sales) AS total_sales
                FROM fact_transactions
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            value = float(
                df.iloc[0]["total_sales"] or 0
            )

            answer_text = (
                f"Total recorded sales are "
                f"{value:,.2f}."
            )

        # ----------------------------------------------------
        # Store ranking
        # ----------------------------------------------------

        elif intent == "store_performance":

            query = """
                SELECT
                    store_id,
                    Store,
                    SUM(Weekly_Sales) AS total_sales
                FROM fact_transactions
                GROUP BY store_id, Store
                ORDER BY total_sales DESC
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            if df.empty:

                answer_text = (
                    "No store sales data was found."
                )

            else:

                row = df.iloc[0]

                answer_text = (
                    f"Store {int(row['Store'])} has the "
                    f"highest total sales with "
                    f"{row['total_sales']:,.2f}."
                )

        # ----------------------------------------------------
        # Department ranking
        # ----------------------------------------------------

        elif intent == "department_performance":

            query = """
                SELECT
                    product_id,
                    Dept,
                    SUM(Weekly_Sales) AS total_sales
                FROM fact_transactions
                GROUP BY product_id, Dept
                ORDER BY total_sales DESC
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            if df.empty:

                answer_text = (
                    "No department sales data was found."
                )

            else:

                row = df.iloc[0]

                answer_text = (
                    f"Department {int(row['Dept'])} has the "
                    f"highest total sales with "
                    f"{row['total_sales']:,.2f}."
                )

        # ----------------------------------------------------
        # Store-specific sales
        # ----------------------------------------------------

        elif intent == "store_sales":

            match = re.search(
                r"store\s*(\d+)",
                question,
                re.IGNORECASE
            )

            if match:

                store_number = int(
                    match.group(1)
                )

                store_id = (
                    f"S{store_number:03d}"
                )

                query = """
                    SELECT
                        store_id,
                        Store,
                        SUM(Weekly_Sales) AS total_sales
                    FROM fact_transactions
                    WHERE store_id = ?
                    GROUP BY store_id, Store
                """

                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[store_id]
                )

                if df.empty:

                    answer_text = (
                        f"No data was found for "
                        f"Store {store_number}."
                    )

                else:

                    value = float(
                        df.iloc[0]["total_sales"]
                    )

                    answer_text = (
                        f"Store {store_number} has total "
                        f"sales of {value:,.2f}."
                    )

            else:

                query = """
                    SELECT
                        store_id,
                        Store,
                        SUM(Weekly_Sales) AS total_sales
                    FROM fact_transactions
                    GROUP BY store_id, Store
                    ORDER BY total_sales DESC
                """

                df = pd.read_sql_query(
                    query,
                    conn
                )

                answer_text = (
                    "Here is the store sales ranking."
                )

        # ----------------------------------------------------
        # Department-specific sales
        # ----------------------------------------------------

        elif intent == "department_sales":

            match = re.search(
                r"(?:department|dept)\s*(\d+)",
                question,
                re.IGNORECASE
            )

            if match:

                dept_number = int(
                    match.group(1)
                )

                product_id = (
                    f"D{dept_number:03d}"
                )

                query = """
                    SELECT
                        product_id,
                        Dept,
                        SUM(Weekly_Sales) AS total_sales
                    FROM fact_transactions
                    WHERE product_id = ?
                    GROUP BY product_id, Dept
                """

                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[product_id]
                )

                if df.empty:

                    answer_text = (
                        f"No data was found for "
                        f"Department {dept_number}."
                    )

                else:

                    value = float(
                        df.iloc[0]["total_sales"]
                    )

                    answer_text = (
                        f"Department {dept_number} has total "
                        f"sales of {value:,.2f}."
                    )

            else:

                query = """
                    SELECT
                        product_id,
                        Dept,
                        SUM(Weekly_Sales) AS total_sales
                    FROM fact_transactions
                    GROUP BY product_id, Dept
                    ORDER BY total_sales DESC
                """

                df = pd.read_sql_query(
                    query,
                    conn
                )

                answer_text = (
                    "Here is the department sales ranking."
                )

        # ----------------------------------------------------
        # Monthly sales
        # ----------------------------------------------------

        elif intent == "monthly_sales":

            query = """
                SELECT
                    strftime('%Y-%m', Date) AS month,
                    SUM(Weekly_Sales) AS total_sales
                FROM fact_transactions
                GROUP BY month
                ORDER BY month
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            answer_text = (
                "Here is the monthly sales breakdown."
            )

        # ----------------------------------------------------
        # Holiday sales
        # ----------------------------------------------------

        elif intent == "holiday_sales":

            query = """
                SELECT
                    IsHoliday,
                    COUNT(*) AS observations,
                    SUM(Weekly_Sales) AS total_sales,
                    AVG(Weekly_Sales) AS average_sales
                FROM fact_transactions
                GROUP BY IsHoliday
                ORDER BY IsHoliday
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            if df.empty:

                answer_text = (
                    "No holiday sales data was found."
                )

            else:

                holiday_sales = 0.0
                non_holiday_sales = 0.0

                for _, row in df.iterrows():

                    if int(row["IsHoliday"]) == 1:
                        holiday_sales = float(
                            row["total_sales"]
                        )
                    else:
                        non_holiday_sales = float(
                            row["total_sales"]
                        )

                answer_text = (
                    f"Holiday sales were "
                    f"{holiday_sales:,.2f}, while "
                    f"non-holiday sales were "
                    f"{non_holiday_sales:,.2f}."
                )

        # ----------------------------------------------------
        # Promotion / markdown
        # ----------------------------------------------------

        elif intent == "promotion_sales":

            query = """
                SELECT
                    promotion_flag,
                    COUNT(*) AS observations,
                    SUM(Weekly_Sales) AS total_sales,
                    AVG(Weekly_Sales) AS average_sales
                FROM fact_transactions
                GROUP BY promotion_flag
                ORDER BY promotion_flag
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            answer_text = (
                "Here is the sales breakdown for "
                "promoted and non-promoted observations."
            )

        # ----------------------------------------------------
        # Data summary
        # ----------------------------------------------------

        elif intent == "data_summary":

            query = """
                SELECT
                    COUNT(*) AS rows_count,
                    COUNT(DISTINCT Store) AS stores,
                    COUNT(DISTINCT Dept) AS departments,
                    MIN(Date) AS start_date,
                    MAX(Date) AS end_date,
                    SUM(Weekly_Sales) AS total_sales
                FROM fact_transactions
            """

            df = pd.read_sql_query(
                query,
                conn
            )

            row = df.iloc[0]

            answer_text = (
                f"The dataset contains "
                f"{int(row['rows_count']):,} weekly observations "
                f"across {int(row['stores'])} stores and "
                f"{int(row['departments'])} departments, "
                f"covering {row['start_date']} to "
                f"{row['end_date']}. "
                f"Total recorded sales are "
                f"{row['total_sales']:,.2f}."
            )

        else:

            df = pd.DataFrame()

            answer_text = (
                "SQL analysis could not be completed."
            )

        return {
            "tool": "SQL",
            "reason": (
                f"SQL analytics selected for the "
                f"'{intent}' business question."
            ),
            "answer": answer_text,
            "data": df,
            "sources": []
        }

    finally:

        conn.close()


# ============================================================
# FORECAST TOOL
# ============================================================

def run_forecast(question):

    # --------------------------------------------------------
    # Department extraction
    # --------------------------------------------------------

    dept_match = re.search(
        r"(?:department|dept)\s*(\d+)",
        question,
        re.IGNORECASE
    )

    # --------------------------------------------------------
    # Store extraction
    # --------------------------------------------------------

    store_match = re.search(
        r"store\s*(\d+)",
        question,
        re.IGNORECASE
    )

    # --------------------------------------------------------
    # Explicit number of weeks
    # --------------------------------------------------------

    weeks_match = re.search(
        r"(\d+)\s*weeks?",
        question,
        re.IGNORECASE
    )

    if not dept_match:

        return {
            "tool": "FORECAST",
            "reason": (
                "The forecast request does not specify "
                "a department."
            ),
            "answer": (
                "Please specify a department, for example "
                "'Department 1'."
            ),
            "data": None,
            "sources": []
        }

    if not store_match:

        return {
            "tool": "FORECAST",
            "reason": (
                "The forecast request does not specify "
                "a store."
            ),
            "answer": (
                "Please specify a store, for example "
                "'Store 1'."
            ),
            "data": None,
            "sources": []
        }

    department_number = int(
        dept_match.group(1)
    )

    store_number = int(
        store_match.group(1)
    )

    product_id = (
        f"D{department_number:03d}"
    )

    store_id = (
        f"S{store_number:03d}"
    )

    # --------------------------------------------------------
    # Determine horizon correctly
    # --------------------------------------------------------

    q = question.lower()

    if weeks_match:

        horizon = int(
            weeks_match.group(1)
        )

    elif "next week" in q:

        horizon = 1

    elif "next month" in q:

        horizon = 4

    else:

        horizon = 4

    # Safety limit
    horizon = max(
        1,
        min(horizon, 52)
    )

    # --------------------------------------------------------
    # Run forecast
    # --------------------------------------------------------

    result = forecast(
        product_id=product_id,
        store_id=store_id,
        horizon=horizon,
        db_path="retailiq.db"
    )

    # --------------------------------------------------------
    # Build clean answer
    # --------------------------------------------------------

    if horizon == 1:

        heading = (
            f"Sales forecast for Department "
            f"{department_number} in Store "
            f"{store_number}:"
        )

    else:

        heading = (
            f"Sales forecast for Department "
            f"{department_number} in Store "
            f"{store_number} for the next "
            f"{horizon} weeks:"
        )

    lines = [heading]

    for _, row in result.iterrows():

        date = pd.to_datetime(
            row["week_start"]
        ).strftime("%Y-%m-%d")

        quantity = float(
            row["forecast_quantity"]
        )

        lines.append(
            f"{date}: {quantity:,.2f} units"
        )

    return {
        "tool": "FORECAST",
        "reason": (
            "Forecast tool selected because the "
            "question asks for future demand."
        ),
        "answer": "\n".join(lines),
        "data": result,
        "sources": []
    }


# ============================================================
# RETRIEVAL DETECTION
# ============================================================

def is_retrieval_question(question):

    q = question.lower()

    retrieval_words = [
        "policy",
        "policies",
        "sop",
        "procedure",
        "procedures",
        "supplier terms",
        "supplier",
        "returns policy",
        "return policy",
        "markdown policy",
        "inventory policy",
        "inventory rule",
        "stockout policy",
        "stockout procedure",
        "what does the policy say",
        "according to the policy",
        "according to our policy",
        "internal documentation",
        "internal procedure",
        "how should staff"
    ]

    return any(
        word in q
        for word in retrieval_words
    )


# ============================================================
# RETRIEVAL TOOL
# ============================================================

def run_retrieval(question):

    if retriever is None:

        return {
            "tool": "RETRIEVAL",
            "reason": (
                "Retrieval was selected because the "
                "question concerns internal documentation, "
                "but the RAG system is unavailable."
            ),
            "answer": (
                "The knowledge-base retrieval system is "
                "not available. Please run "
                "'python build_rag.py'."
            ),
            "data": None,
            "sources": []
        }

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    try:

        results = retriever.search(
            question,
            top_k=5
        )

    except Exception as e:

        return {
            "tool": "RETRIEVAL",
            "reason": "Retrieval execution failed.",
            "answer": (
                f"Retrieval failed: "
                f"{type(e).__name__}: {e}"
            ),
            "data": None,
            "sources": []
        }

    if not results:

        return {
            "tool": "RETRIEVAL",
            "reason": (
                "No relevant knowledge-base evidence "
                "was retrieved."
            ),
            "answer": (
                "I could not find relevant information "
                "in the RetailIQ knowledge base."
            ),
            "data": None,
            "sources": []
        }

    # --------------------------------------------------------
    # Normalize retrieved results
    # --------------------------------------------------------

    evidence = []

    for item in results:

        if not isinstance(
            item,
            dict
        ):
            continue

        content = str(
            item.get(
                "content",
                item.get(
                    "text",
                    ""
                )
            )
        ).strip()

        if not content:
            continue

        source = str(
            item.get(
                "source",
                item.get(
                    "document",
                    "knowledge_base.csv"
                )
            )
        )

        title = str(
            item.get(
                "document_title",
                source
            )
        )

        section = str(
            item.get(
                "section",
                ""
            )
        )

        topic = str(
            item.get(
                "topic",
                ""
            )
        )

        score = float(
            item.get(
                "score",
                0
            )
        )

        evidence.append(
            {
                "content": content,
                "source": source,
                "title": title,
                "section": section,
                "topic": topic,
                "score": score
            }
        )

    if not evidence:

        return {
            "tool": "RETRIEVAL",
            "reason": (
                "Retrieved records did not contain "
                "usable evidence."
            ),
            "answer": (
                "Relevant records were found, but no "
                "usable policy text was available."
            ),
            "data": None,
            "sources": []
        }

    # --------------------------------------------------------
    # Remove duplicate documents
    # --------------------------------------------------------

    unique_evidence = []

    seen = set()

    for item in evidence:

        key = (
            item["source"],
            item["section"],
            item["content"]
        )

        if key in seen:
            continue

        seen.add(key)

        unique_evidence.append(
            item
        )

    evidence = unique_evidence

    # --------------------------------------------------------
    # Select the strongest relevant evidence
    #
    # Prefer the best result from each document, but keep
    # the context concise.
    # --------------------------------------------------------

    evidence.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    primary = evidence[0]

    selected = [
        primary
    ]

    primary_source = primary["source"]

    # Add another chunk only if it is from the same
    # document or has a very strong retrieval score.
    for item in evidence[1:]:

        if item["source"] == primary_source:

            selected.append(
                item
            )

        elif (
            len(selected) < 3
            and item["score"] >= primary["score"] * 0.90
        ):

            selected.append(
                item
            )

        if len(selected) >= 3:
            break

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for item in selected:

        context_parts.append(
            f"""
SOURCE: {item['source']}
DOCUMENT: {item['title']}
SECTION: {item['section']}
TOPIC: {item['topic']}

{item['content']}
""".strip()
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Gemini synthesis
    # --------------------------------------------------------

    answer_text = None

    if gemini_model is not None:

        prompt = f"""
You are the RetailIQ internal business assistant.

Answer the user's question using ONLY the supplied
RetailIQ knowledge-base evidence.

USER QUESTION:
{question}

KNOWLEDGE-BASE EVIDENCE:
{context}

RULES:
1. Do not invent facts.
2. Do not use outside knowledge.
3. Do not reproduce the entire retrieved document.
4. Give a concise professional answer.
5. Use 2 to 5 sentences unless a formula or short
   bullet list improves clarity.
6. Preserve important policy rules and formulas.
7. If required information is unavailable, explicitly
   say that it is unavailable.
8. Do not mention irrelevant evidence.
9. These are synthetic RetailIQ project policies,
   not actual Walmart internal policies.
10. Cite the most relevant document.
11. If a section is relevant, include the section name.

Use this exact citation format at the end:

Source: <document filename> | Section: <section>

Return only the final answer.
"""

        try:

            response = gemini_model.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            answer_text = getattr(
                response,
                "text",
                None
            )

        except Exception as e:

            print(
                f"Gemini RAG synthesis warning: {e}"
            )

    # --------------------------------------------------------
    # Fallback without Gemini
    # --------------------------------------------------------

    if not answer_text:

        answer_text = (
            f"{primary['content']}\n\n"
            f"Source: {primary['source']} "
            f"| Section: {primary['section']}"
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = []

    for item in selected:

        citation = item["source"]

        if item["section"]:

            citation = (
                f"{citation} | "
                f"Section: {item['section']}"
            )

        if citation not in sources:

            sources.append(
                citation
            )

    return {
        "tool": "RETRIEVAL",
        "reason": (
            "Retrieval tool selected because the "
            "question concerns internal RetailIQ "
            "policies or procedures."
        ),
        "answer": answer_text.strip(),
        "data": None,
        "sources": sources
    }


# ============================================================
# TOOL PLANNER
# ============================================================

def choose_tool(question):

    # --------------------------------------------------------
    # Forecast first
    # --------------------------------------------------------

    if is_forecast_question(
        question
    ):

        return (
            "forecast",
            "The question asks for future demand or sales."
        )

    # --------------------------------------------------------
    # Retrieval second
    # --------------------------------------------------------

    if is_retrieval_question(
        question
    ):

        return (
            "retrieval",
            "The question concerns internal policies or procedures."
        )

    # --------------------------------------------------------
    # SQL third
    # --------------------------------------------------------

    sql_intent = detect_sql_intent(
        question
    )

    if sql_intent is not None:

        return (
            "sql",
            f"The question requests historical database analytics: {sql_intent}."
        )

    # --------------------------------------------------------
    # Gemini planner for ambiguous questions
    # --------------------------------------------------------

    if gemini_model is not None:

        prompt = f"""
You are the planner for RetailIQ.

Choose exactly ONE tool:

SQL
FORECAST
RETRIEVAL

SQL:
Historical sales, totals, rankings, store analysis,
department analysis, monthly analysis, holidays,
promotions, and database metrics.

FORECAST:
Future demand or sales predictions.

RETRIEVAL:
Internal policies, SOPs, supplier terms,
inventory rules, markdown rules, returns procedures,
and internal documentation.

Question:
{question}

Return only:
SQL
FORECAST
or
RETRIEVAL
"""

        try:

            response = gemini_model.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            selected = (
                response.text
                .strip()
                .upper()
            )

            if selected in {
                "SQL",
                "FORECAST",
                "RETRIEVAL"
            }:

                return (
                    selected.lower(),
                    "Gemini planner selected the appropriate tool."
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Safe default
    # --------------------------------------------------------

    return (
        "sql",
        "Defaulted to SQL analytics."
    )


# ============================================================
# FORECAST QUESTION DETECTION
# ============================================================

def is_forecast_question(question):

    q = question.lower()

    forecast_terms = [
        "forecast",
        "predict",
        "prediction",
        "future sales",
        "future demand",
        "next week",
        "next weeks",
        "next month",
        "expected sales",
        "expected demand",
        "demand forecast",
        "sales forecast",
        "forecast sales"
    ]

    return any(
        term in q
        for term in forecast_terms
    )


# ============================================================
# MAIN AGENT
# ============================================================

def answer(question):

    if not question or not str(question).strip():

        return {
            "tool": "SYSTEM",
            "reason": "Empty question.",
            "answer": (
                "Please enter a business question."
            ),
            "data": None,
            "sources": []
        }

    question = str(
        question
    ).strip()

    try:

        tool, planner_reason = choose_tool(
            question
        )

        # ----------------------------------------------------
        # Execute selected tool
        # ----------------------------------------------------

        if tool == "forecast":

            result = run_forecast(
                question
            )

        elif tool == "retrieval":

            result = run_retrieval(
                question
            )

        else:

            result = run_sql(
                question
            )

        # ----------------------------------------------------
        # Make sure reasoning is always visible
        # ----------------------------------------------------

        if not result.get(
            "reason"
        ):

            result["reason"] = planner_reason

        return result

    except Exception as e:

        return {
            "tool": "SYSTEM",
            "reason": "Execution error.",
            "answer": (
                f"Execution failed: "
                f"{type(e).__name__}: {e}"
            ),
            "data": None,
            "sources": []
        }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [

        "What are the total sales?",

        "Which store has the highest total sales?",

        "Which department has the highest total sales?",

        "What are the sales for Store 1?",

        "What are the sales for Department 1?",

        "Show monthly sales.",

        "How did holidays affect sales?",

        "Predict the next 4 weeks for Department 1 in Store 1",

        "Predict the next week sales for Department 1 in Store 2",

        "What is the inventory reorder policy?",

        "When should a slow-moving product be marked down?",

        "What is the procedure for damaged stock?",

        "How should returns be treated in demand modelling?"
    ]

    print(
        "\nRetailIQ Agent Test"
    )

    print(
        "=" * 70
    )

    for question in test_questions:

        print(
            f"\nUSER: {question}"
        )

        result = answer(
            question
        )

        print(
            f"TOOL: {result['tool']}"
        )

        print(
            f"REASON: {result['reason']}"
        )

        print(
            "ANSWER:"
        )

        print(
            result["answer"]
        )

        if result.get("sources"):

            print(
                "SOURCES:"
            )

            for source in result["sources"]:

                print(
                    f"  - {source}"
                )

        print(
            "-" * 70
        )