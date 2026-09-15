# ============================================================
# RetailIQ - Streamlit Application
# Walmart Dataset Version
# ============================================================

from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import RetailIQAgent
from forecast import forecast_next_weeks


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetailIQ",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📦 RetailIQ")

st.caption(
    "Retail Sales Intelligence • Demand Forecasting • "
    "AI Business Assistant"
)


# ============================================================
# DATABASE
# ============================================================

DB_PATH = Path(
    "retailiq.db"
)

if not DB_PATH.exists():

    st.error(
        "retailiq.db was not found."
    )

    st.code(
        "python pipeline.py\n"
        "python train_models.py"
    )

    st.stop()


# ============================================================
# LOAD DATABASE
# ============================================================

@st.cache_data
def load_database():

    with sqlite3.connect(
        DB_PATH
    ) as connection:

        products = pd.read_sql(
            """
            SELECT *
            FROM dim_product
            """,
            connection
        )

        stores = pd.read_sql(
            """
            SELECT *
            FROM dim_store
            """,
            connection
        )

        weekly = pd.read_sql(
            """
            SELECT *
            FROM fact_weekly_sales
            """,
            connection
        )

    weekly[
        "week_start"
    ] = pd.to_datetime(
        weekly[
            "week_start"
        ],
        errors="coerce"
    )

    return (
        products,
        stores,
        weekly
    )


products, stores, weekly = (
    load_database()
)


# ============================================================
# AGENT
# ============================================================

agent = RetailIQAgent()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Executive Dashboard",
        "🔮 Forecast Explorer",
        "🤖 AI Assistant"
    ]
)


# ============================================================
# TAB 1
# EXECUTIVE DASHBOARD
# ============================================================

with tab1:

    st.header(
        "📊 Executive Dashboard"
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_sales = (
        weekly["quantity"]
        .sum()
    )

    total_revenue = (
        weekly["revenue"]
        .sum()
    )

    department_count = (
        products[
            "product_id"
        ]
        .nunique()
    )

    store_count = (
        stores[
            "store_id"
        ]
        .nunique()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        "Total Revenue",
        f"{total_revenue:,.0f}"
    )

    col3.metric(
        "Departments",
        f"{department_count:,}"
    )

    col4.metric(
        "Stores",
        f"{store_count:,}"
    )

    st.divider()

    # --------------------------------------------------------
    # Weekly trend
    # --------------------------------------------------------

    trend = (
        weekly
        .groupby(
            "week_start",
            as_index=False
        )
        .agg(
            sales=(
                "quantity",
                "sum"
            ),
            revenue=(
                "revenue",
                "sum"
            )
        )
    )

    fig = px.line(
        trend,
        x="week_start",
        y="revenue",
        markers=True,
        title="Weekly Revenue Trend"
    )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Revenue",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Department + Store charts
    # --------------------------------------------------------

    col1, col2 = (
        st.columns(2)
    )

    # --------------------------------------------------------
    # Departments
    # --------------------------------------------------------

    with col1:

        department_sales = (
            weekly
            .merge(
                products[
                    [
                        "product_id",
                        "product_name"
                    ]
                ],
                on="product_id",
                how="left"
            )
            .groupby(
                [
                    "product_id",
                    "product_name"
                ],
                as_index=False
            )
            .agg(
                sales=(
                    "quantity",
                    "sum"
                )
            )
            .sort_values(
                "sales",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            department_sales,
            x="sales",
            y="product_name",
            orientation="h",
            title="Top 10 Departments"
        )

        fig.update_layout(
            xaxis_title="Sales",
            yaxis_title="Department"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Stores
    # --------------------------------------------------------

    with col2:

        store_sales = (
            weekly
            .merge(
                stores[
                    [
                        "store_id",
                        "store_name"
                    ]
                ],
                on="store_id",
                how="left"
            )
            .groupby(
                [
                    "store_id",
                    "store_name"
                ],
                as_index=False
            )
            .agg(
                sales=(
                    "quantity",
                    "sum"
                )
            )
            .sort_values(
                "sales",
                ascending=False
            )
        )

        fig = px.bar(
            store_sales,
            x="store_name",
            y="sales",
            title="Store Performance"
        )

        fig.update_layout(
            xaxis_title="Store",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Markdown analysis
    # --------------------------------------------------------

    st.subheader(
        "📈 Markdown / Promotion Analysis"
    )

    if "promotion_rate" in weekly.columns:

        markdown_data = (
            weekly
            .assign(
                status=weekly[
                    "promotion_rate"
                ]
                .apply(
                    lambda x:
                    "Markdown"
                    if x > 0
                    else "No Markdown"
                )
            )
            .groupby(
                "status",
                as_index=False
            )
            .agg(
                average_sales=(
                    "quantity",
                    "mean"
                )
            )
        )

        markdown_data[
            "average_sales"
        ] = (
            markdown_data[
                "average_sales"
            ]
            .round(2)
        )

        fig = px.bar(
            markdown_data,
            x="status",
            y="average_sales",
            title="Average Weekly Sales: Markdown vs No Markdown"
        )

        fig.update_layout(
            xaxis_title="Status",
            yaxis_title="Average Weekly Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    metrics_path = Path(
        "artifacts/metrics.csv"
    )

    if metrics_path.exists():

        st.subheader(
            "🤖 Model Performance"
        )

        metrics = pd.read_csv(
            metrics_path
        )

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2
# FORECAST EXPLORER
# ============================================================

with tab2:

    st.header(
        "🔮 Demand Forecast Explorer"
    )

    st.write(
        "Select a department and store to "
        "forecast future weekly demand."
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    # --------------------------------------------------------
    # Department
    # --------------------------------------------------------

    with col1:

        product_options = (
            products[
                "product_id"
            ]
            .tolist()
        )

        selected_product = (
            st.selectbox(
                "Department",
                product_options
            )
        )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    with col2:

        store_options = (
            stores[
                "store_id"
            ]
            .tolist()
        )

        selected_store = (
            st.selectbox(
                "Store",
                store_options
            )
        )

    # --------------------------------------------------------
    # Horizon
    # --------------------------------------------------------

    with col3:

        horizon = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=12,
            value=4
        )

    # --------------------------------------------------------
    # Selected department information
    # --------------------------------------------------------

    product_row = products[
        products[
            "product_id"
        ]
        == selected_product
    ]

    if not product_row.empty:

        st.info(
            f"Selected Department: "
            f"{product_row.iloc[0]['product_name']}"
        )

    # --------------------------------------------------------
    # Selected store
    # --------------------------------------------------------

    store_row = stores[
        stores[
            "store_id"
        ]
        == selected_store
    ]

    if not store_row.empty:

        st.info(
            f"Selected Store: "
            f"{store_row.iloc[0]['store_name']}"
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    if st.button(
        "🔮 Generate Forecast",
        type="primary"
    ):

        try:

            forecast_result = (
                forecast_next_weeks(
                    weekly,
                    selected_product,
                    selected_store,
                    horizon
                )
            )

            st.success(
                "Forecast generated successfully."
            )

            # ------------------------------------------------
            # Clean table
            # ------------------------------------------------

            forecast_table = (
                forecast_result
                .copy()
            )

            if "week_start" in forecast_table.columns:

                forecast_table[
                    "week_start"
                ] = pd.to_datetime(
                    forecast_table[
                        "week_start"
                    ],
                    errors="coerce"
                ).dt.strftime(
                    "%Y-%m-%d"
                )

            for column in forecast_table.columns:

                if (
                    column != "week_start"
                    and pd.api.types.is_numeric_dtype(
                        forecast_table[column]
                    )
                ):

                    forecast_table[column] = (
                        forecast_table[column]
                        .round(2)
                    )

            # ------------------------------------------------
            # Rename columns
            # ------------------------------------------------

            rename_map = {}

            if "week_start" in forecast_table.columns:

                rename_map[
                    "week_start"
                ] = "Week"

            if "predicted_demand" in forecast_table.columns:

                rename_map[
                    "predicted_demand"
                ] = "Predicted Demand"

            forecast_table = (
                forecast_table
                .rename(
                    columns=rename_map
                )
            )

            # ------------------------------------------------
            # Result table
            # ------------------------------------------------

            st.subheader(
                "Forecast Results"
            )

            st.dataframe(
                forecast_table,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            if (
                "week_start"
                in forecast_result.columns
                and
                "predicted_demand"
                in forecast_result.columns
            ):

                chart_data = (
                    forecast_result
                    .copy()
                )

                chart_data[
                    "week_start"
                ] = pd.to_datetime(
                    chart_data[
                        "week_start"
                    ],
                    errors="coerce"
                )

                fig = px.line(
                    chart_data,
                    x="week_start",
                    y="predicted_demand",
                    markers=True,
                    title=(
                        "Future Demand Forecast"
                    )
                )

                fig.update_layout(
                    xaxis_title="Week",
                    yaxis_title="Predicted Demand",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Forecast could not be generated: {error}"
            )


# ============================================================
# TAB 3
# AI ASSISTANT
# ============================================================

with tab3:

    st.header(
        "🤖 Ask RetailIQ"
    )

    st.write(
        "Ask questions about sales, departments, "
        "stores, markdowns, forecasts, or "
        "internal business policies."
    )

    # --------------------------------------------------------
    # Gemini status
    # --------------------------------------------------------

    if agent.gemini.enabled:

        st.success(
            "🟢 Gemini planner is enabled"
        )

    else:

        st.warning(
            "🟡 Gemini planner is unavailable. "
            "RetailIQ is using the fallback planner."
        )

    # --------------------------------------------------------
    # Example questions
    # --------------------------------------------------------

    st.markdown(
        "**Example questions:**"
    )

    examples = [
        "Which departments have the highest sales?",
        "Which store has the highest sales?",
        "Show monthly sales.",
        "What are the markdown effects?",
        "Forecast Department 1 for Store 1 for 4 weeks.",
        "What is the return policy?"
    ]

    for example in examples:

        st.caption(
            "• " + example
        )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    question = st.text_input(
        "Ask a business question",
        placeholder=(
            "Example: Which store has the highest sales?"
        )
    )

    # --------------------------------------------------------
    # Ask
    # --------------------------------------------------------

    if st.button(
        "Ask RetailIQ",
        type="primary"
    ) and question:

        with st.spinner(
            "RetailIQ is analyzing your question..."
        ):

            result = agent.answer(
                question
            )

        # ----------------------------------------------------
        # Tool selection
        # ----------------------------------------------------

        st.subheader(
            "🧠 Tool Selection"
        )

        tool = result.get(
            "tool",
            "unknown"
        )

        reason = result.get(
            "reason",
            ""
        )

        tool_names = {

            "sql":
                "🗄️ SQL Analytics",

            "forecast":
                "🔮 Forecasting",

            "retrieval":
                "📚 Document Retrieval",

            "system":
                "⚙️ System",

            "none":
                "❌ None"
        }

        st.info(
            f"**Selected Tool:** "
            f"{tool_names.get(tool, tool)}"
        )

        st.caption(
            f"**Why:** {reason}"
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.subheader(
            "💡 Answer"
        )

        answer = result.get(
            "answer",
            ""
        )

        data = result.get(
            "data",
            None
        )

        # ----------------------------------------------------
        # Display text answer
        # ----------------------------------------------------

        if answer:

            st.write(
                answer
            )

        # ----------------------------------------------------
        # Display structured table
        # ----------------------------------------------------

        if isinstance(
            data,
            pd.DataFrame
        ):

            display_data = (
                data.copy()
            )

            # ------------------------------------------------
            # Format dates
            # ------------------------------------------------

            for column in display_data.columns:

                if (
                    "date" in column.lower()
                    or "week" in column.lower()
                ):

                    try:

                        display_data[
                            column
                        ] = pd.to_datetime(
                            display_data[
                                column
                            ],
                            errors="coerce"
                        ).dt.strftime(
                            "%Y-%m-%d"
                        )

                    except Exception:

                        pass

            # ------------------------------------------------
            # Format numbers
            # ------------------------------------------------

            for column in display_data.columns:

                if pd.api.types.is_numeric_dtype(
                    display_data[column]
                ):

                    display_data[
                        column
                    ] = (
                        display_data[
                            column
                        ]
                        .round(2)
                    )

            # ------------------------------------------------
            # Rename common columns
            # ------------------------------------------------

            rename_map = {

                "product_id":
                    "Department ID",

                "product_name":
                    "Department",

                "Dept":
                    "Dept",

                "store_id":
                    "Store ID",

                "store_name":
                    "Store",

                "Store":
                    "Store Number",

                "sales":
                    "Sales",

                "total_sales":
                    "Total Sales",

                "average_weekly_sales":
                    "Avg Weekly Sales",

                "total_revenue":
                    "Total Revenue",

                "total_units":
                    "Total Units",

                "promotion_status":
                    "Promotion Status",

                "average_markdown":
                    "Average Markdown",

                "maximum_markdown":
                    "Maximum Markdown"
            }

            display_data = (
                display_data
                .rename(
                    columns=rename_map
                )
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            []
        )

        if sources:

            st.subheader(
                "📚 Sources"
            )

            for source in sources:

                st.caption(
                    f"• {source}"
                )