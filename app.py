import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import RetailIQAgent
from forecast import forecast_next_weeks

st.set_page_config(
    page_title="RetailIQ",
    page_icon="📦",
    layout="wide"
)

st.title("📦 RetailIQ")
st.caption("Demand Forecasting and Multi-Tool Business Assistant")

DB = Path("retailiq.db")

if not DB.exists():
    st.error(
        "Database not found. Run:\n\n"
        "python generate_demo_data.py\n"
        "python pipeline.py\n"
        "python train_models.py"
    )
    st.stop()

agent = RetailIQAgent()

with sqlite3.connect(DB) as con:
    products = pd.read_sql(
        "SELECT * FROM dim_product",
        con
    )
    stores = pd.read_sql(
        "SELECT * FROM dim_store",
        con
    )
    weekly = pd.read_sql(
        "SELECT * FROM fact_weekly_sales",
        con
    )

tab1, tab2, tab3 = st.tabs([
    "Executive Dashboard",
    "Forecast Explorer",
    "AI Assistant"
])

with tab1:
    st.subheader("Executive Dashboard")

    total_revenue = weekly["revenue"].sum()
    total_units = weekly["quantity"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", f"{total_revenue:,.0f}")
    c2.metric("Units", f"{total_units:,.0f}")
    c3.metric(
        "Products",
        f"{products['product_id'].nunique():,}"
    )

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"]
    )

    trend = weekly.groupby(
        "week_start",
        as_index=False
    ).agg(
        revenue=("revenue", "sum"),
        units=("quantity", "sum")
    )

    st.plotly_chart(
        px.line(
            trend,
            x="week_start",
            y="revenue",
            title="Revenue Trend"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        cat = (
            weekly.merge(products, on="product_id")
            .groupby("category", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )

        st.plotly_chart(
            px.bar(
                cat,
                x="category",
                y="revenue",
                title="Category Performance"
            ),
            use_container_width=True
        )

    with col2:
        rank = (
            weekly.merge(stores, on="store_id")
            .groupby(
                ["store_id", "store_name"],
                as_index=False
            )["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )

        st.plotly_chart(
            px.bar(
                rank,
                x="store_name",
                y="revenue",
                title="Store Ranking"
            ),
            use_container_width=True
        )

with tab2:
    st.subheader("Forecast Explorer")

    p = st.selectbox(
        "Product",
        products["product_id"].tolist()
    )
    s = st.selectbox(
        "Store",
        stores["store_id"].tolist()
    )
    h = st.slider(
        "Forecast horizon (weeks)",
        1, 12, 4
    )

    if st.button("Generate Forecast"):
        try:
            fc = forecast_next_weeks(
                weekly, p, s, h
            )

            st.dataframe(
                fc,
                use_container_width=True
            )

            st.plotly_chart(
                px.line(
                    fc,
                    x="week_start",
                    y="predicted_demand",
                    markers=True,
                    title=f"{p} at {s} — Forecast"
                ),
                use_container_width=True
            )

        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("Business Assistant")

    question = st.text_input(
        "Ask a business question",
        placeholder=(
            "Examples: What are the top 5 products by revenue? "
            "What is the returns policy? "
            "Forecast P001 at S01 for 4 weeks."
        )
    )

    if st.button("Ask RetailIQ") and question:
        result = agent.answer(question)

        st.info(
            f"Tool selected: **{result['tool']}**"
        )
        st.caption(
            f"Selection reasoning: {result['reason']}"
        )

        if isinstance(result["answer"], pd.DataFrame):
            st.dataframe(
                result["answer"],
                use_container_width=True
            )
        else:
            st.write(result["answer"])

        if result.get("sql"):
            with st.expander("Generated SQL"):
                st.code(
                    result["sql"],
                    language="sql"
                )

        if result.get("sources"):
            st.caption(
                "Sources: " +
                ", ".join(result["sources"])
            )
