from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))
from modeling import load_and_train, build_live_feature_row, fetch_market_data, FEATURE_COLUMNS

st.set_page_config(
    page_title="NIFTY 50 Correction-Risk Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NIFTY 50 Correction-Risk Monitor")
st.caption("Historical analysis • Model comparison • Experimental market-data prediction")

DATA_PATH = APP_DIR / "data" / "historical_nifty50.csv"

@st.cache_resource(show_spinner="Training models on the historical dataset...")
def get_artifacts():
    return load_and_train(DATA_PATH)

try:
    artifacts = get_artifacts()
except Exception as exc:
    st.error(f"Could not prepare the models: {exc}")
    st.stop()

data = artifacts["data"]
metrics = artifacts["metrics"]
test_predictions = artifacts["test_predictions"]
models = artifacts["models"]
thresholds = artifacts["thresholds"]

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        ["Overview", "Historical analysis", "Live-data prediction", "Model performance", "Downloads"],
    )
    st.divider()
    st.caption("Model is trained from the included 2010–2020 labeled dataset.")
    st.caption("Live-data mode depends on Yahoo Finance data availability and is experimental.")

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical observations", f"{len(data):,}")
    c2.metric("Coverage", f"{data['date'].min():%d %b %Y} – {data['date'].max():%d %b %Y}")
    c3.metric("Features used", str(len(FEATURE_COLUMNS)))
    c4.metric("Models", "2 ML + VIX rule")
    st.markdown(
        """
        This dashboard explores whether market indicators can classify the supplied
        `correction_within_20d` label. Logistic Regression and Random Forest are trained
        chronologically; an India VIX threshold is included as a simple benchmark.
        """
    )
    st.warning(
        "Research prototype only. The target labels are taken from the supplied dataset; "
        "their exact generation formula and real-world correctness have not been independently verified. "
        "The historical dataset ends on 31 December 2020."
    )
    st.subheader("Historical NIFTY 50 close")
    fig = px.line(data, x="date", y="nifty50_close", title="NIFTY 50 closing level (provided dataset)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Latest available historical row")
    st.dataframe(data.tail(1), use_container_width=True, hide_index=True)

elif page == "Historical analysis":
    st.subheader("Historical market analysis")
    left, right = st.columns(2)
    with left:
        fig = px.line(data, x="date", y="nifty50_close", title="NIFTY 50 close")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(data, x="date", y="india_vix_close", title="India VIX close")
        st.plotly_chart(fig, use_container_width=True)

    if "nifty_return_20d_recalc" in data:
        fig = px.line(data, x="date", y="nifty_return_20d_recalc", title="NIFTY 20-observation return")
        st.plotly_chart(fig, use_container_width=True)

    counts = data["correction_within_20d"].value_counts().rename(index={0: "No correction label", 1: "Correction label"})
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Supplied target label", "y": "Observations"},
                 title="Distribution of supplied target labels")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Explore a date range**")
    start, end = st.date_input(
        "Date range",
        value=(data["date"].min().date(), data["date"].max().date()),
        min_value=data["date"].min().date(),
        max_value=data["date"].max().date(),
    )
    filtered = data[(data["date"].dt.date >= start) & (data["date"].dt.date <= end)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

elif page == "Live-data prediction":
    st.subheader("Experimental latest-market-data prediction")
    st.info(
        "This is not an exchange-grade real-time feed. The app attempts to retrieve recent daily "
        "quotes from Yahoo Finance. Quotes may be delayed, incomplete, unavailable, or revised. "
        "The model's historical training period ends in 2020."
    )
    if st.button("Fetch latest market data and score", type="primary"):
        with st.spinner("Fetching recent market data..."):
            market = fetch_market_data()
        if market is None or market.empty:
            st.error("Market data could not be retrieved. Try again later or use the historical-analysis section.")
        else:
            feature_row, diagnostics = build_live_feature_row(market, artifacts["training_medians"])
            st.write("**Latest retrieved market observations**")
            st.dataframe(market.tail(5), use_container_width=True)
            if diagnostics:
                st.warning("Unavailable current inputs were filled using training-set medians: " + ", ".join(diagnostics))
            result_cols = st.columns(2)
            for col, model_name in zip(result_cols, ["Logistic Regression", "Random Forest"]):
                model = models[model_name]
                probability = float(model.predict_proba(feature_row[FEATURE_COLUMNS])[:, 1][0])
                threshold = thresholds[model_name]
                prediction = int(probability >= threshold)
                with col:
                    st.metric(f"{model_name} score", f"{probability:.1%}")
                    st.write(f"Validation-selected threshold: **{threshold:.3f}**")
                    st.write("**Predicted supplied-label class:** " + ("Correction-risk class" if prediction else "No-correction-risk class"))
            st.caption(
                "The probability is a model score for the dataset's supplied label, not a calibrated "
                "probability of an actual market crash. Do not use it as investment advice."
            )
            st.download_button(
                "Download latest feature row (CSV)",
                feature_row.to_csv(index=False).encode("utf-8"),
                file_name="latest_market_feature_row.csv",
                mime="text/csv",
            )
    else:
        st.write("Press the button to fetch the latest available daily observations.")

elif page == "Model performance":
    st.subheader("Chronological holdout performance")
    st.caption("Metrics below are recomputed by this app using the deployment feature set.")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    metric = st.selectbox("Metric to compare", ["precision", "recall", "f1", "accuracy", "pr_auc", "roc_auc"])
    if metric in metrics.columns:
        fig = px.bar(metrics, x="model", y=metric, title=f"Model comparison: {metric}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrices")
    for model_name, cm in artifacts["confusion_matrices"].items():
        st.markdown(f"**{model_name}**")
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            labels={"x": "", "y": "", "color": "Count"},
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Random Forest feature importance")
    importance = artifacts["feature_importance"]
    fig = px.bar(importance.head(15).sort_values("importance"), x="importance", y="feature",
                 orientation="h", title="Top feature importances")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Downloads":
    st.subheader("Download project outputs")
    downloads = {
        "Historical dataset": (data.to_csv(index=False), "historical_dataset.csv"),
        "Recomputed model metrics": (metrics.to_csv(index=False), "model_metrics.csv"),
        "Test-period predictions": (test_predictions.to_csv(index=False), "test_predictions.csv"),
        "Random Forest feature importance": (artifacts["feature_importance"].to_csv(index=False), "feature_importance.csv"),
    }
    for label, (content, filename) in downloads.items():
        st.download_button(f"Download {label}", content.encode("utf-8"), filename, "text/csv", key=filename)

st.divider()
st.caption("Educational research prototype • Not investment advice • Validate labels, inputs, and data licensing before operational use.")
