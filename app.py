import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

from agent import answer
from forecast import forecast, load_model_metrics

st.set_page_config(page_title="RetailIQ", page_icon="📊", layout="wide")

DB="retailiq.db"

st.title("RetailIQ")
st.caption("Demand Forecasting and Multi-Tool Business Assistant")


def q(sql, params=()):
    conn=sqlite3.connect(DB)
    try: return pd.read_sql_query(sql, conn, params=params)
    finally: conn.close()


tabs=st.tabs(["Executive Dashboard","Forecast Explorer","AI Business Assistant"])

with tabs[0]:
    st.header("Executive Dashboard")
    summary=q("""SELECT COUNT(*) rows, COUNT(DISTINCT Store) stores,
                        COUNT(DISTINCT Dept) departments,
                        SUM(Weekly_Sales) sales
                 FROM fact_transactions""").iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Weekly observations",f"{int(summary.rows):,}")
    b.metric("Stores",int(summary.stores))
    c.metric("Departments",int(summary.departments))
    d.metric("Total sales",f"{summary.sales:,.0f}")

    monthly=q("""SELECT strftime('%Y-%m',Date) month,SUM(Weekly_Sales) sales
                 FROM fact_transactions GROUP BY month ORDER BY month""")
    st.plotly_chart(px.line(monthly,x="month",y="sales",title="Monthly Sales"),use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        stores=q("""SELECT Store,SUM(Weekly_Sales) sales FROM fact_transactions
                    GROUP BY Store ORDER BY sales DESC""")
        st.plotly_chart(px.bar(stores.head(10),x="Store",y="sales",title="Top Stores"),use_container_width=True)
    with c2:
        depts=q("""SELECT Dept,SUM(Weekly_Sales) sales FROM fact_transactions
                   GROUP BY Dept ORDER BY sales DESC""")
        st.plotly_chart(px.bar(depts.head(10),x="Dept",y="sales",title="Top Departments"),use_container_width=True)

    promo=q("""SELECT promotion_flag,AVG(Weekly_Sales) avg_sales
               FROM fact_transactions GROUP BY promotion_flag""")
    promo["period"]=promo["promotion_flag"].map({0:"No markdown",1:"Markdown proxy"})
    st.plotly_chart(px.bar(promo,x="period",y="avg_sales",title="Promotional/Markdown Proxy Effect"),use_container_width=True)

    st.info("Walmart source data does not provide inventory levels or true selling price. Those requirements are supported by the schema and optional input files, but are not fabricated from unavailable fields.")

with tabs[1]:
    st.header("Forecast Explorer")
    stores=q("SELECT DISTINCT Store FROM fact_transactions ORDER BY Store")["Store"].tolist()
    depts=q("SELECT DISTINCT Dept FROM fact_transactions ORDER BY Dept")["Dept"].tolist()
    c1,c2,c3=st.columns(3)
    store=c1.selectbox("Store",stores)
    dept=c2.selectbox("Department",depts)
    horizon=c3.slider("Forecast horizon (weeks)",1,12,4)

    if st.button("Generate Forecast"):
        result=forecast(f"D{int(dept):03d}",f"S{int(store):03d}",horizon,DB)
        st.dataframe(result,use_container_width=True)
        st.plotly_chart(px.line(result,x="week_start",y="forecast_quantity",
                                 markers=True,title="Forecast"),use_container_width=True)

    metrics=load_model_metrics()
    if metrics is not None:
        st.subheader("Model validation")
        st.dataframe(metrics,use_container_width=True)
        st.caption("MAPE is calculated on non-zero/meaningful actuals to avoid division-by-zero distortion.")

with tabs[2]:
    st.header("AI Business Assistant")
    st.write("Ask about sales, stores, departments, forecasts, promotions, or internal policies.")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("tool"):
                st.caption(f"Tool: {m['tool']} · Reason: {m['reason']}")
            if m.get("sources"):
                st.caption("Sources: "+", ".join(m["sources"]))

    question=st.chat_input("Ask RetailIQ...")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        result=answer(question)
        st.session_state.messages.append({
            "role":"assistant","content":result["answer"],
            "tool":result["tool"],"reason":result["reason"],
            "sources":result.get("sources",[])
        })
        st.rerun()
