import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Energy Price Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_results():
    results_path = Path("results/benchmark_results.json")
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


@st.cache_data
def load_data():
    data_path = Path("data/energy_prices.csv")
    if not data_path.exists():
        return None
    return pd.read_csv(data_path, parse_dates=["timestamp"])


def render_sidebar(results):
    st.sidebar.title("Dashboard Controls")
    available_models = [r["model_name"] for r in results]
    selected_models = st.sidebar.multiselect(
        "Select models to compare",
        available_models,
        default=available_models,
    )
    metric_options = ["MAE", "RMSE", "MAPE", "sMAPE", "R2"]
    primary_metric = st.sidebar.selectbox("Primary metric", metric_options, index=0)
    show_horizon = st.sidebar.checkbox("Show per-horizon analysis", value=True)
    show_predictions = st.sidebar.checkbox("Show prediction plots", value=True)
    forecast_samples = st.sidebar.slider("Prediction samples to display", 50, 500, 200)
    return selected_models, primary_metric, show_horizon, show_predictions, forecast_samples


def render_overview(results, selected_models):
    st.header("Model Performance Overview")

    filtered = [r for r in results if r["model_name"] in selected_models]
    if not filtered:
        st.warning("No models selected.")
        return

    cols = st.columns(len(filtered))
    for i, r in enumerate(filtered):
        m = r["overall_metrics"]
        with cols[i]:
            st.subheader(r["model_name"])
            st.metric("MAE", f"{m['MAE']:.4f}")
            st.metric("RMSE", f"{m['RMSE']:.4f}")
            st.metric("R2", f"{m['R2']:.4f}")
            st.metric("Training Time", f"{m['training_time']:.2f}s")


def render_comparison_chart(results, selected_models, primary_metric):
    st.header("Model Comparison")

    filtered = [r for r in results if r["model_name"] in selected_models]
    if not filtered:
        return

    metrics_list = ["MAE", "RMSE", "MAPE", "sMAPE", "R2"]

    comparison_data = []
    for r in filtered:
        for metric in metrics_list:
            comparison_data.append({
                "Model": r["model_name"],
                "Metric": metric,
                "Value": r["overall_metrics"][metric],
            })
    df = pd.DataFrame(comparison_data)

    col1, col2 = st.columns(2)

    with col1:
        primary_df = df[df["Metric"] == primary_metric]
        fig = px.bar(
            primary_df, x="Model", y="Value",
            color="Model", title=f"Comparison by {primary_metric}",
            text_auto=".4f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rows = []
        for r in filtered:
            row = {"Model": r["model_name"]}
            row.update(r["overall_metrics"])
            rows.append(row)
        table_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(table_df.style.highlight_min(
            subset=["MAE", "RMSE", "MAPE", "sMAPE", "training_time"], color="lightgreen"
        ).highlight_max(
            subset=["R2"], color="lightgreen"
        ), use_container_width=True)

    normalized_data = []
    for metric in metrics_list:
        metric_vals = df[df["Metric"] == metric]["Value"].values
        if metric_vals.max() - metric_vals.min() > 0:
            for r in filtered:
                val = r["overall_metrics"][metric]
                if metric == "R2":
                    norm_val = (val - metric_vals.min()) / (metric_vals.max() - metric_vals.min())
                else:
                    norm_val = 1 - (val - metric_vals.min()) / (metric_vals.max() - metric_vals.min())
                normalized_data.append({
                    "Model": r["model_name"],
                    "Metric": metric,
                    "Score": norm_val,
                })

    if normalized_data:
        norm_df = pd.DataFrame(normalized_data)
        fig = px.line_polar(
            norm_df, r="Score", theta="Metric", color="Model",
            line_close=True, title="Normalized Performance Radar",
            range_r=[0, 1.1],
        )
        fig.update_traces(fill="toself", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)


def render_horizon_analysis(results, selected_models):
    st.header("Per-Horizon Forecast Analysis")

    filtered = [r for r in results if r["model_name"] in selected_models]
    if not filtered:
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for r in filtered:
            steps = [m["horizon_step"] for m in r["horizon_metrics"]]
            mae_vals = [m["MAE"] for m in r["horizon_metrics"]]
            fig.add_trace(go.Scatter(
                x=steps, y=mae_vals, mode="lines+markers", name=r["model_name"],
            ))
        fig.update_layout(
            title="MAE by Forecast Horizon Step",
            xaxis_title="Horizon Step (hours ahead)",
            yaxis_title="MAE",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for r in filtered:
            steps = [m["horizon_step"] for m in r["horizon_metrics"]]
            rmse_vals = [m["RMSE"] for m in r["horizon_metrics"]]
            fig.add_trace(go.Scatter(
                x=steps, y=rmse_vals, mode="lines+markers", name=r["model_name"],
            ))
        fig.update_layout(
            title="RMSE by Forecast Horizon Step",
            xaxis_title="Horizon Step (hours ahead)",
            yaxis_title="RMSE",
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for r in filtered:
        steps = [m["horizon_step"] for m in r["horizon_metrics"]]
        r2_vals = [m["R2"] for m in r["horizon_metrics"]]
        fig.add_trace(go.Scatter(
            x=steps, y=r2_vals, mode="lines+markers", name=r["model_name"],
        ))
    fig.update_layout(
        title="R2 Score by Forecast Horizon Step",
        xaxis_title="Horizon Step (hours ahead)",
        yaxis_title="R2",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_predictions(results, selected_models, n_samples):
    st.header("Prediction vs Actual")

    filtered = [r for r in results if r["model_name"] in selected_models]
    if not filtered:
        return

    actuals = np.array(filtered[0]["actuals"])[:n_samples, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actuals, mode="lines", name="Actual",
        line={"color": "black", "width": 2},
    ))
    for r in filtered:
        preds = np.array(r["predictions"])[:n_samples, 0]
        fig.add_trace(go.Scatter(
            y=preds, mode="lines", name=r["model_name"], opacity=0.7,
        ))
    fig.update_layout(
        title="1-Step Ahead Predictions (first horizon step)",
        xaxis_title="Sample Index",
        yaxis_title="Price (scaled)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    model_select = st.selectbox(
        "Select model for detailed error analysis",
        [r["model_name"] for r in filtered],
    )

    selected_result = next(r for r in filtered if r["model_name"] == model_select)
    preds = np.array(selected_result["predictions"])[:n_samples, 0]
    errors = actuals - preds

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            x=errors, nbins=50,
            title=f"Error Distribution - {model_select}",
            labels={"x": "Prediction Error"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            x=actuals, y=preds,
            title=f"Actual vs Predicted - {model_select}",
            labels={"x": "Actual", "y": "Predicted"},
        )
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Prediction",
            line={"dash": "dash", "color": "red"},
        ))
        st.plotly_chart(fig, use_container_width=True)


def render_training_history(results, selected_models):
    st.header("Training History (Neural Network Models)")

    filtered = [
        r for r in results
        if r["model_name"] in selected_models and r.get("training_history")
    ]
    if not filtered:
        st.info("No training history available for selected models.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for r in filtered:
            hist = r["training_history"]
            if "loss" in hist:
                fig.add_trace(go.Scatter(
                    y=hist["loss"], mode="lines", name=f"{r['model_name']} - train",
                ))
            if "val_loss" in hist:
                fig.add_trace(go.Scatter(
                    y=hist["val_loss"], mode="lines", name=f"{r['model_name']} - val",
                    line={"dash": "dash"},
                ))
        fig.update_layout(
            title="Training Loss Curves",
            xaxis_title="Epoch",
            yaxis_title="Loss (MSE)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for r in filtered:
            hist = r["training_history"]
            if "mae" in hist:
                fig.add_trace(go.Scatter(
                    y=hist["mae"], mode="lines", name=f"{r['model_name']} - train",
                ))
            if "val_mae" in hist:
                fig.add_trace(go.Scatter(
                    y=hist["val_mae"], mode="lines", name=f"{r['model_name']} - val",
                    line={"dash": "dash"},
                ))
        fig.update_layout(
            title="Training MAE Curves",
            xaxis_title="Epoch",
            yaxis_title="MAE",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_data_exploration(df):
    st.header("Dataset Exploration")

    if df is None:
        st.warning("No dataset found.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{len(df):,}")
    col2.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
    col3.metric("Mean Price", f"{df['price'].mean():.2f}")
    col4.metric("Price Std Dev", f"{df['price'].std():.2f}")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            df.set_index("timestamp").resample("D")["price"].mean().reset_index(),
            x="timestamp", y="price",
            title="Daily Average Energy Price",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="hour", y="price",
            title="Price Distribution by Hour of Day",
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        numeric_cols = ["price", "temperature", "demand", "wind_generation", "solar_generation", "gas_price"]
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Demand vs Price", "Wind Generation vs Price"))
        fig.add_trace(
            go.Scatter(
                x=df["demand"], y=df["price"], mode="markers",
                marker={"size": 2, "opacity": 0.3}, name="Demand",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["wind_generation"], y=df["price"], mode="markers",
                marker={"size": 2, "opacity": 0.3}, name="Wind",
            ),
            row=2, col=1,
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Energy Price Forecasting - ML Benchmark Dashboard")
    st.markdown("Compare Random Forest, XGBoost, DNN, and LSTM models for energy price prediction.")

    results = load_results()
    df = load_data()

    if results is None:
        st.error(
            "No benchmark results found. Run the training pipeline first:\n\n"
            "```bash\nuv run python src/pipelines/training.py\n```"
        )
        if df is not None:
            render_data_exploration(df)
        return

    selected_models, primary_metric, show_horizon, show_predictions, forecast_samples = render_sidebar(results)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Comparison", "Predictions", "Training History", "Data Exploration",
    ])

    with tab1:
        render_overview(results, selected_models)

    with tab2:
        render_comparison_chart(results, selected_models, primary_metric)
        if show_horizon:
            render_horizon_analysis(results, selected_models)

    with tab3:
        if show_predictions:
            render_predictions(results, selected_models, forecast_samples)

    with tab4:
        render_training_history(results, selected_models)

    with tab5:
        render_data_exploration(df)


if __name__ == "__main__":
    main()
