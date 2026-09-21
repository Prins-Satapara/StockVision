import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.company_info import COMPANY_INFO
from src.config import FEATURES, RESULTS_DIR, STOCKS
from src.data_loader import load_stock_data
from src.prediction import predict_stock

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout="wide",
)

HORIZONS = {
    "1D": {"days": 1, "label": "next trading day"},
    "5D": {"days": 5, "label": "next 5 trading days"},
    "20D": {"days": 20, "label": "next 20 trading days"},
}

# Column names used in results/final_model_selection.csv (written by notebook 09)
TEST_COLUMNS = {
    "Logistic Regression": "Logistic Test Accuracy",
    "Default Random Forest": "Default RF Test Accuracy",
    "Tuned Random Forest": "Tuned RF Test Accuracy",
}
CV_COLUMNS = {
    "Logistic Regression": "Logistic CV",
    "Default Random Forest": "Default RF CV",
    "Tuned Random Forest": "Tuned RF CV",
}

TIME_RANGES = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
    "ALL": None,
}


# ---------------------------------------------------------------------------
# Cached data loaders (so the app does not reload files on every click)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(stock):
    return load_stock_data(stock)


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_selection():
    return pd.read_csv(RESULTS_DIR / "final_model_selection.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def get_prediction(stock, horizon):
    return predict_stock(stock, horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def get_baseline(stock, horizon):
    """
    Accuracy of the laziest possible strategy on the test period:
    always predict the class (UP/DOWN) that was more common in the training period.
    It uses the same 80/20 chronological split as the notebooks.
    """
    days = HORIZONS[horizon]["days"]
    data = get_stock_data(stock)

    future_close = data["Close"].shift(-days)
    target = (future_close > data["Close"]).astype(int)

    features = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    valid = future_close.notna() & features.notna().all(axis=1)

    y = target[valid].reset_index(drop=True)
    split = int(len(y) * 0.8)
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    majority = 1 if y_train.mean() >= 0.5 else 0
    return {
        "accuracy": float((y_test == majority).mean()),
        "label": "UP" if majority == 1 else "DOWN",
    }


def money(value, currency):
    return f"{currency}{value:,.2f}"


def style_chart(fig, height, y_title):
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------
st.title("StockVision Dashboard")
st.write("Cross-market stock analysis and machine learning prediction")
st.caption("Built with Python, Streamlit, Plotly, scikit-learn, Optuna and Yahoo Finance")

ctrl1, ctrl2 = st.columns([1, 2])
with ctrl1:
    selected_stock = st.selectbox("Select stock", STOCKS)
with ctrl2:
    horizon = st.segmented_control(
        "Prediction horizon",
        options=list(HORIZONS.keys()),
        default="1D",
        format_func=lambda h: {"1D": "1 day", "5D": "5 days", "20D": "20 days"}[h],
    ) or "1D"  # segmented_control returns None if the user un-selects it

try:
    df = get_stock_data(selected_stock)
    selection = get_model_selection()
except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Run the pipeline (notebooks 01 to 09) to generate the data, models and results first."
    )
    st.stop()

info = COMPANY_INFO.get(selected_stock, {})
currency = "₹" if info.get("Country") == "India" else "$"
if info:
    st.caption(f"{info['Company']} · {info['Sector']} · {info['Exchange']} · prices in {currency}")

# ---------------------------------------------------------------------------
# Market overview
# ---------------------------------------------------------------------------
st.subheader(f"{selected_stock} overview")

latest_close = df["Close"].iloc[-1]
previous_close = df["Close"].iloc[-2]
price_change = latest_close - previous_close
price_change_pct = price_change / previous_close * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest close", money(latest_close, currency))
m2.metric("Daily change", f"{price_change:+,.2f}", f"{price_change_pct:+.2f}%")
m3.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
m4.metric("Data as of", df["Date"].iloc[-1].strftime("%d %b %Y"))

# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------
st.subheader("Stock price history")

c1, c2 = st.columns(2)
with c1:
    time_range = st.segmented_control(
        "Time range", options=list(TIME_RANGES.keys()), default="1Y"
    ) or "1Y"
with c2:
    chart_type = st.segmented_control(
        "Chart type", options=["Line", "Candlestick"], default="Line"
    ) or "Line"

indicators = st.multiselect(
    "Technical indicators", options=["SMA 20", "SMA 50", "EMA 20"], default=[]
)

latest_date = df["Date"].max()
offset = TIME_RANGES[time_range]
start_date = df["Date"].min() if offset is None else latest_date - offset
chart_df = df[df["Date"] > start_date].copy()

price_fig = go.Figure()
if chart_type == "Line":
    price_fig.add_trace(
        go.Scatter(x=chart_df["Date"], y=chart_df["Close"], mode="lines", name="Close")
    )
else:
    price_fig.add_trace(
        go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name=selected_stock,
        )
    )

indicator_columns = {
    "SMA 20": ("SMA_20", "#F47BD2", "dash"),
    "SMA 50": ("SMA_50", "#5C0505", "dash"),
    "EMA 20": ("EMA_20", "#E3C74B", "dash"),
}
for name in indicators:
    column, color, dash = indicator_columns[name]
    price_fig.add_trace(
        go.Scatter(
            x=chart_df["Date"], y=chart_df[column], mode="lines", name=name, line=dict(color=color, dash=dash)
        )
    )

st.plotly_chart(style_chart(price_fig, 550, f"Price ({currency})"), width="stretch")

# ---------------------------------------------------------------------------
# Technical analysis
# ---------------------------------------------------------------------------
st.subheader("Technical analysis")

st.markdown("#### Relative Strength Index (RSI)")
rsi_fig = go.Figure()
rsi_fig.add_trace(
    go.Scatter(
        x=chart_df["Date"],
        y=chart_df["RSI_14"],
        mode="lines",
        name="RSI 14",
        line=dict(color="#2d75b9", width=1.5),
    )
)
rsi_fig.add_hline(
    y=70, line_dash="dash", line_color="#5fe385", line_width=2,
    annotation_text="Overbought", annotation_position="top left",
    annotation_font=dict(size=13, color="#c7c7c7"),
)
rsi_fig.add_hline(
    y=30, line_dash="dash", line_color="#d73d0e", line_width=2,
    annotation_text="Oversold", annotation_position="bottom left",
    annotation_font=dict(size=13, color="#c7c7c7"),
)
rsi_fig.update_yaxes(range=[0, 100])
rsi_fig.update_layout(showlegend=False)
st.plotly_chart(style_chart(rsi_fig, 400, "RSI 14"), width="stretch")

st.markdown("#### Moving Average Convergence Divergence (MACD)")
macd_fig = go.Figure()
macd_fig.add_trace(
    go.Bar(
        x=chart_df["Date"],
        y=chart_df["MACD_Hist"],
        name="Histogram",
        marker_color=["#16A34A" if v >= 0 else "#EF4444" for v in chart_df["MACD_Hist"]],
        opacity=0.6,
    )
)
macd_fig.add_trace(
    go.Scatter(
        x=chart_df["Date"], y=chart_df["MACD"], mode="lines", name="MACD",
        line=dict(color="#4f9de6", width=1.8),
    )
)
macd_fig.add_trace(
    go.Scatter(
        x=chart_df["Date"], y=chart_df["MACD_Signal"], mode="lines", name="Signal",
        line=dict(color="#F59E0B", width=1.5, dash="dash"),
    )
)
macd_fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
macd_fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(style_chart(macd_fig, 400, "MACD"), width="stretch")

# ---------------------------------------------------------------------------
# ML prediction
# ---------------------------------------------------------------------------
st.divider()
st.header("ML prediction")

horizon_label = HORIZONS[horizon]["label"]
row = selection[(selection["Stock"] == selected_stock) & (selection["Horizon"] == horizon)]

if row.empty:
    st.warning(f"No trained model found for {selected_stock} at the {horizon} horizon.")
else:
    row = row.iloc[0]
    model_name = row["Best Model"]
    test_accuracy = row[TEST_COLUMNS[model_name]]
    cv_accuracy = row[CV_COLUMNS[model_name]]
    baseline = get_baseline(selected_stock, horizon)

    latest_features = df[FEATURES].iloc[-1]
    if not np.isfinite(latest_features.astype(float)).all():
        st.warning(
            "The latest row has missing or invalid feature values, so no prediction "
            "can be shown. Re-run the data and feature engineering steps."
        )
    else:
        try:
            prediction = get_prediction(selected_stock, horizon)
        except FileNotFoundError:
            prediction = None
            st.error(
                f"Model file for {selected_stock} ({horizon}) is missing in models/final/. "
                "Run notebook 09 to create it."
            )

        if prediction is not None:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Model leans", prediction["Prediction"])
            p2.metric("Model probability", f"{prediction['Probability']:.1f}%")
            p3.metric("Model used", model_name)
            p4.metric("Based on data of", prediction["Date"])

            direction_text = "UP" if prediction["Prediction"] == "UP" else "DOWN"

            st.markdown(
                f"For the **{horizon_label}**, the model predicts **{direction_text}** "
                f"with **{prediction['Probability']:.1f}% model confidence**."
            )

            beats_baseline = test_accuracy > baseline["accuracy"]
            comparison = (
                f"On held-out test data this model scored **{test_accuracy:.1%}**. "
                f"Always guessing {baseline['label']} (the more common outcome in the "
                f"training period) would have scored **{baseline['accuracy']:.1%}**."
            )
            if beats_baseline:
                st.info(comparison + " The model did better than that simple guess on this test period.")
            else:
                st.warning(
                    comparison + " The model did **not** beat that simple guess, "
                    "so treat this prediction as very weak."
                )

            st.caption(
                "Model probability is the model's own estimate, not a guarantee. "
                "For educational use only. This is not financial advice."
            )

    # -----------------------------------------------------------------------
    # Model performance
    # -----------------------------------------------------------------------
    st.divider()
    st.header("Model performance")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Selected model", model_name)
    q2.metric("Test accuracy", f"{test_accuracy:.1%}")
    q3.metric("Cross-validation accuracy", f"{cv_accuracy:.1%}")
    q4.metric("Simple-guess baseline", f"{baseline['accuracy']:.1%}")

    st.markdown("### Model comparison")
    models = list(TEST_COLUMNS.keys())
    comparison_fig = go.Figure()
    comparison_fig.add_trace(
        go.Bar(
            name="Test accuracy",
            y=models,
            x=[row[TEST_COLUMNS[m]] for m in models],
            orientation="h",
            text=[f"{row[TEST_COLUMNS[m]]:.1%}" for m in models],
            textposition="outside",
        )
    )
    comparison_fig.add_trace(
        go.Bar(
            name="Cross-validation accuracy",
            y=models,
            x=[row[CV_COLUMNS[m]] for m in models],
            orientation="h",
            text=[f"{row[CV_COLUMNS[m]]:.1%}" for m in models],
            textposition="outside",
        )
    )
    comparison_fig.add_vline(
        x=baseline["accuracy"], line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Always {baseline['label']}", annotation_position="top",
    )
    comparison_fig.update_layout(
        barmode="group",
        height=380,
        xaxis_title="Accuracy",
        yaxis_title="",
        xaxis=dict(range=[0, 0.85], tickformat=".0%"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
    )
    st.plotly_chart(comparison_fig, width="stretch")
    st.caption(
        "The selected model is the one with the best cross-validation accuracy. "
        "The saved model was retrained on all available data, so the test score above "
        "comes from the version trained on the first 80% of the history."
    )

    with st.expander(f"All stocks at the {horizon} horizon"):
        summary = selection[selection["Horizon"] == horizon].copy()
        summary["Test accuracy"] = summary.apply(
            lambda r: r[TEST_COLUMNS[r["Best Model"]]], axis=1
        )
        summary["Simple-guess baseline"] = summary["Stock"].apply(
            lambda s: get_baseline(s, horizon)["accuracy"]
        )
        summary["Beats baseline"] = np.where(
            summary["Test accuracy"] > summary["Simple-guess baseline"], "Yes", "No"
        )
        summary = summary.rename(
            columns={"Best Model": "Selected model", "Best CV Accuracy": "CV accuracy"}
        )[["Stock", "Selected model", "CV accuracy", "Test accuracy",
           "Simple-guess baseline", "Beats baseline"]]
        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
            column_config={
                "CV accuracy": st.column_config.NumberColumn(format="percent"),
                "Test accuracy": st.column_config.NumberColumn(format="percent"),
                "Simple-guess baseline": st.column_config.NumberColumn(format="percent"),
            },
        )

st.divider()
st.caption(
    "**StockVision** | Machine learning based stock analysis dashboard. "
    "Built with Python, Streamlit, Plotly, scikit-learn, Optuna and Yahoo Finance. "
    "Educational purpose only. This application is not financial advice."
)