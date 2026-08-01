import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.config import STOCKS, RESULTS_DIR
from src.data_loader import load_stock_data
from src.prediction import predict_stock

# configure the page
st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout='wide'
)

# dashboard header
st.title("StockVision Dashboard")

st.write(
    "Cross-Market Stock Analysis & Machine Learning Prediction"
)

st.subheader("Built With")

st.write(
    """
    • Python
    • Streamlit
    • Plotly
    • Scikit-learn
    • Optuna
    • Yahoo Finance
    """
)
# stock selecter
selected_stock = st.selectbox("Select Stock", STOCKS)

df = load_stock_data(selected_stock)

comparison_df = pd.read_csv(
    RESULTS_DIR / "final_model_selection.csv"
)

selected_model = comparison_df.loc[
    comparison_df["Stock"] == selected_stock
].iloc[0]

prediction = predict_stock(selected_stock)

st.subheader(f"{selected_stock} OverView")

col1, col2, col3, col4, col5 = st.columns(5)


latest_close = df["Close"].iloc[-1]
previous_close = df["Close"].iloc[-2]
price_change = latest_close - previous_close

price_change_pct = (price_change / previous_close) * 100

latest_volume = df["Volume"].iloc[-1]

with col1:
    st.metric(
        "Latest Close",
        f"{latest_close:.2f}"
    )
    
with col2:
    st.metric(
        "Daily Change", 
        f"{price_change:+,.2f}",
        f"{price_change_pct:+.2f}%"
    )
    
with col3:
    st.metric(
        "Volume", 
        f"{latest_volume:,.0f}"
    )
    
with col4:
    st.metric(
        "ML Prediction",
        prediction["Prediction"]
    )

with col5:
    st.metric(
        "Prediction Confidence",
        f'{prediction["Probability"]:.2f}%'
    )

##************Chart Section************

st.subheader("Stock Price History")

time_range = st.segmented_control(
    "Time Range",
    options=["1M", "3M", "6M", "1Y", "5Y", "ALL"],
    default="1Y"
)

latest_date = df['Date'].max()

if time_range == "1M":
    start_date = latest_date - pd.DateOffset(months=1)
elif time_range == "3M":
    start_date = latest_date - pd.DateOffset(months=3)
elif time_range == "6M":
    start_date = latest_date - pd.DateOffset(months=6)
elif time_range == "1Y":
    start_date = latest_date - pd.DateOffset(years=1)
elif time_range == "5Y":
    start_date = latest_date - pd.DateOffset(years=5)
else:
    start_date = df["Date"].min()
    
chart_df = df[df["Date"] > start_date].copy()

chart_type = st.segmented_control(
    "Chart Type", 
    options=["Line", "Candlestick"],
    default="Line"
)

indicators = st.multiselect(
    "Technical Indicators", 
    options=["SMA 20", "SMA 50", "EMA 20"],
    default=[]
)

fig = go.Figure()
if chart_type == "Line":
    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["Close"],
            mode="lines",
            name="Latest Close"
        )
    )
else:
    fig.add_trace(
        go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name=selected_stock
        )
    )

if "SMA 20" in indicators:
    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["SMA_20"],
            mode="lines",
            name="SMA 20",
            line_color="#F59E0B"
        )
    )
if "SMA 50" in indicators:
    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["SMA_50"],
            mode="lines",
            name="SMA 50",
            line_color="#8B5CF6"
        )
    )
if "EMA 20" in indicators:
    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["EMA_20"],
            mode="lines", 
            name="EMA 20",
            line_color="#10B981"
        )
    )


fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    height=550
)

fig.update_xaxes(
    rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

#......... Technical Analysis .........
st.subheader("Technical Analysis")

st.markdown("#### Relative Strength Index (RSI)")

rsi_fig = go.Figure()

rsi_fig.add_trace(
    go.Scatter(
        x=chart_df["Date"],
        y=chart_df["RSI_14"],
        mode="lines",
        name= "RSI 14",
        line=dict(color="#2d75b9", width=1.5)
    )
)
rsi_fig.add_hline(
    y=70,
    line_dash="dash",
    line_color="#5fe385",
    line_width=2,
    annotation_text="Overbought",
    annotation_position="top left",
    annotation_font=dict(size=13, color="#c7c7c7")
)

rsi_fig.add_hline(
    y=30,
    line_dash="dash",
    line_color="#d73d0e",
    line_width=2,
    annotation_text="Oversold",
    annotation_position="bottom left",
    annotation_font=dict(size=13, color="#c7c7c7")
)


rsi_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI 14",
    hovermode="x unified",
    height=450,
    showlegend=False,
)

rsi_fig.update_yaxes(
    range=[0, 100]
)

st.plotly_chart(
    rsi_fig,
    use_container_width=True
)


st.markdown("#### Moving Average Convergence Divergence (MACD)")

macd_fig = go.Figure()

macd_fig.add_trace(
    go.Bar(
        x=chart_df["Date"],
        y=chart_df["MACD_Hist"],
        name="Histogram",
        marker_color=[
            "#16A34A" if v >= 0 else "#EF4444" for v in chart_df["MACD_Hist"]
        ],
        opacity=0.6
    )
)

macd_fig.add_trace(
    go.Scatter(
        x=chart_df["Date"],
        y=chart_df["MACD"],
        mode="lines",
        name="MACD",
        line=dict(color="#4f9de6", width=1.8)
    )
)

macd_fig.add_trace(
    go.Scatter(
        x=chart_df["Date"],
        y=chart_df["MACD_Signal"],
        mode="lines",
        name="Signal",
        line=dict(color="#F59E0B", width=1.5, dash="dash")
    )
)

macd_fig.add_hline(
    y=0,
    line_dash="dot",
    line_color="gray",
    line_width=1
)

macd_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="MACD",
    hovermode="x unified",
    height=450,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    margin=dict(r=40)
)

st.plotly_chart(
    macd_fig,
    use_container_width=True
)


#............ AI Prediction ..............#
st.divider()

st.header("ML Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Prediction",
        prediction["Prediction"]
    )

with col2:
    st.metric(
        "Confidence",
        f'{prediction["Probability"]:.2f}%'
    )
    
col3, col4 = st.columns(2)

with col3:
    st.metric(
        "Latest Trading Date",
        prediction["Date"]
    )

with col4:
    st.metric(
        "Selected Model",
        selected_model["Best Model"]
    )
if prediction["Prediction"] == "UP":

    st.success(
        f"The model predicts an UPWARD movement with {prediction['Probability']:.2f}% confidence."
    )

else:

    st.error(
        f"The model predicts a DOWNWARD movement with {prediction['Probability']:.2f}% confidence."
    )
st.info(
    """
    This prediction is generated using the trained Random Forest model
    based on historical price data and technical indicators.
    It should be used for educational purposes only and should not be
    considered financial advice.
    """
)
    
#........ Model Perfomance......#
st.divider()
st.header("Model Performance")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Best Model",
        selected_model["Best Model"]
    )
with col2:
    st.metric(
        "Best Accuracy",
        f"{selected_model['Best Accuracy']:.2%}"
    )


st.markdown("### Model Comparison")

comparison = pd.DataFrame({
    "Model" : ["Logistic Regression", "Default Random Forest", "Tuned Random Forest"],
    "Accuracy" : [selected_model["Logistic Regression"], selected_model["Default Random Forest"], selected_model["Tuned Random Forest"]]
})

fig = px.bar(
    comparison,
    x="Accuracy",
    y="Model",
    orientation="h",
    text="Accuracy"
)
fig.update_traces(
    texttemplate="%{x:.2%}",
    textposition="outside"
)
fig.update_layout(
    height=300,
    xaxis_title="Accuracy",
    yaxis_title=""
    
)
st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()

st.caption(
    """
    **StockVision** | Machine Learning Based Stock Analysis Dashboard

    Built using **Python**, **Streamlit**, **Plotly**, **Scikit-learn**, **Optuna** and **Yahoo Finance**

    Educational purpose only. This application is not financial advice.
    """
)

