import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import numpy as np

from model import TICKER, load_and_prepare_data, predict_latest, train_models


st.set_page_config(page_title="StockSenseAI", page_icon="📈", layout="wide")

# Custom CSS to style sidebar radio options as modern interactive navigation buttons
st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background-color: #131D2E;
        border: 1px solid #263449;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 2px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1B283F;
        border-color: #00D4AA;
        transform: translateX(2px);
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: rgba(0, 212, 170, 0.12) !important;
        border-color: #00D4AA !important;
        color: #00D4AA !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## 📈 StockSenseAI")
    st.caption("AI-powered stock analysis")

    st.divider()

    st.markdown("### Stock Selection")
    POPULAR_TICKERS = [
        "INFY.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
        "TATAMOTORS.NS", "SBIN.NS", "ITC.NS", "AAPL", "NVDA", "MSFT"
    ]
    ticker_choice = st.selectbox("Select Stock", POPULAR_TICKERS, index=0)
    custom_ticker = st.text_input("Or enter custom ticker:", placeholder="e.g. TSLA, WIPRO.NS")
    selected_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else ticker_choice

    st.caption(f"Active ticker: **{selected_ticker}**")

    st.divider()

    st.markdown("### Dashboard")
    NAV_OPTIONS = [
        "🌐 Full Dashboard",
        "📊 Market Overview",
        "📈 Technical Indicators",
        "🤖 ML Predictions",
        "🎯 Model Performance",
    ]
    selected_view = st.radio(
        "Dashboard Navigation",
        NAV_OPTIONS,
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    st.info(
        "Market data is provided by Yahoo Finance. "
        "Quotes may be delayed."
    )


@st.cache_data(show_spinner="Downloading market data...")
def get_data(ticker: str) -> pd.DataFrame:
    return load_and_prepare_data(ticker)


@st.cache_resource(show_spinner="Training models...")
def get_models(data: pd.DataFrame) -> dict:
    return train_models(data)



def line_chart(
    data: pd.DataFrame,
    title: str,
    series: list[tuple[str, str]],
    currency: str = "₹"
) -> go.Figure:
    figure = go.Figure()

    chart_colors = ["#00D4AA", "#4DA3FF", "#F6C85F", "#B794F4"]

    for index, (column, label) in enumerate(series):
        figure.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data[column],
                name=label,
                mode="lines",
                line=dict(
                    color=chart_colors[index % len(chart_colors)],
                    width=2
                ),
                hovertemplate=f"%{{x|%d %b %Y}}<br>{currency}%{{y:,.2f}}<extra></extra>"
                if "Price" in title or "Closing" in title
                else "%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>"
            )
        )

    figure.update_layout(
        title=dict(text=title, font=dict(size=18, color="#E6EDF7")),
        template="plotly_dark",
        paper_bgcolor="#131D2E",
        plot_bgcolor="#131D2E",
        font=dict(color="#C5D0E0"),
        height=360,
        margin=dict(l=18, r=18, t=60, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#263449",
            zeroline=False
        )
    )

    return figure


currency = "$" if not selected_ticker.endswith((".NS", ".BO")) else "₹"

try:
    data = get_data(selected_ticker)
    models = get_models(data)
except Exception as error:
    st.error(f"Unable to load data or train the models for {selected_ticker}: {error}")
    st.stop()

latest, previous = data.iloc[-1], data.iloc[-2]
prediction = predict_latest(models, data.iloc[[-1]])
daily_change = ((latest["Close"] - previous["Close"]) / previous["Close"]) * 100


# Dashboard header
st.markdown(
    """
    <div style="
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #263449;
        margin-bottom: 20px;
    ">
        <div style="font-size: 14px; color: #00D4AA;
                    font-weight: 600; letter-spacing: 2px;">
            AI-POWERED MARKET ANALYTICS
        </div>
        <h1 style="margin: 6px 0 4px 0; color: #E6EDF7;">
            StockSenseAI 📈
        </h1>
        <div style="color: #93A4BB; font-size: 15px;">
            Technical indicators · Machine learning · Market insights
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"Selected stock: **{selected_ticker}** · Historical data from Yahoo Finance")

# ==========================================
# 📊 Market Overview
# ==========================================
if selected_view in ["📊 Market Overview", "🌐 Full Dashboard"]:
    st.markdown("### 📊 Market Overview")
    summary = st.columns(4)
    summary[0].metric(
        "CLOSING PRICE",
        f"{currency}{latest['Close']:,.2f}"
    )
    summary[1].metric(
        "DAILY CHANGE",
        f"{daily_change:+.2f}%",
        delta=f"{daily_change:+.2f}%"
    )
    summary[2].metric(
        "TRADING VOLUME",
        f"{latest['Volume']:,.0f}"
    )
    summary[3].metric(
        "MODEL PRICE ESTIMATE",
        f"{currency}{prediction['next_close']:,.2f}"
    )
    st.caption(
        "The estimated price is a model output, not a guaranteed future market price."
    )

    st.subheader("Historical Price")
    price_figure = line_chart(
        data,
        "Closing Price with Moving Averages",
        [("Close", "Close"), ("MA_20", "MA20"), ("MA_50", "MA50")],
        currency=currency
    )
    price_figure.update_yaxes(title=f"Price ({currency})")
    st.plotly_chart(price_figure, width="stretch")

    if selected_view == "🌐 Full Dashboard":
        st.divider()

# ==========================================
# 📈 Technical Indicators
# ==========================================
if selected_view in ["📈 Technical Indicators", "🌐 Full Dashboard"]:
    st.markdown("### 📈 Technical Indicators")
    st.caption("Relative Strength Index (RSI) momentum and Moving Average Convergence Divergence (MACD).")
    indicators = st.columns(2)
    with indicators[0]:
        rsi_figure = line_chart(data, "RSI", [("RSI", "RSI")])
        rsi_figure.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_figure.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(rsi_figure, width="stretch")

    with indicators[1]:
        macd_figure = make_subplots(specs=[[{"secondary_y": True}]])
        macd_figure.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], name="MACD"), secondary_y=False)
        macd_figure.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], name="Signal"), secondary_y=False)
        macd_figure.add_trace(go.Bar(x=data["Date"], y=data["MACD_Histogram"], name="Histogram"), secondary_y=True)

        macd_figure.update_layout(
            title="MACD — Momentum Indicator",
            template="plotly_dark",
            paper_bgcolor="#131D2E",
            plot_bgcolor="#131D2E",
            font=dict(color="#C5D0E0"),
            height=360,
            margin=dict(l=18, r=18, t=60, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
            hovermode="x unified"
        )

        macd_figure.update_yaxes(gridcolor="#263449")
        st.plotly_chart(macd_figure, width="stretch")

    if selected_view == "🌐 Full Dashboard":
        st.divider()

# ==========================================
# 🤖 ML Predictions
# ==========================================
if selected_view in ["🤖 ML Predictions", "🌐 Full Dashboard"]:
    st.markdown("### 🤖 ML Predictions")
    st.caption("Machine learning directional predictions and estimated next-day closing price.")
    prediction_cards = st.columns(3)
    prediction_cards[0].metric("SVM prediction", "UP" if prediction["svm"] else "DOWN")
    prediction_cards[1].metric("XGBoost prediction", "UP" if prediction["xgb"] else "DOWN")
    prediction_cards[2].metric("XGBoost predicted next-day close", f"{currency}{prediction['next_close']:,.2f}")

    if selected_view == "🌐 Full Dashboard":
        st.divider()

# ==========================================
# 🎯 Model Performance
# ==========================================
if selected_view in ["🎯 Model Performance", "🌐 Full Dashboard"]:
    st.markdown("### 🎯 Model Performance")
    st.caption("Calculated using the chronological 20% test set.")
    metrics = models["metrics"]
    svm_column, xgb_column, regression_column = st.columns(3)
    for column, title, values in [
        (svm_column, "SVM", metrics["svm"]),
        (xgb_column, "XGBoost Classifier", metrics["xgb"]),
    ]:
        column.markdown(f"#### {title}")
        for metric, value in values.items():
            column.metric(metric, f"{value * 100:.2f}%")

    regression_column.markdown("#### XGBoost Regressor")
    for metric, value in metrics["regression"].items():
        regression_column.metric("R²" if metric == "R2" else metric, f"{value:,.4f}")

    st.subheader("Actual vs Predicted Closing Price")
    st.caption(
        "Comparison of actual market prices and XGBoost predictions "
        "on the chronological test set."
    )

    comparison = pd.DataFrame({
        "Date": pd.to_datetime(models["test_dates"]),
        "Actual": np.asarray(models["actual_close"], dtype=float),
        "Predicted": np.asarray(models["predicted_close"], dtype=float),
    }).reset_index(drop=True)

    # Calculate regression performance.
    errors = comparison["Actual"] - comparison["Predicted"]
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors ** 2).mean()))

    actual_mean = comparison["Actual"].mean()
    ss_total = ((comparison["Actual"] - actual_mean) ** 2).sum()
    ss_residual = (errors ** 2).sum()
    r2 = (
        float(1 - ss_residual / ss_total)
        if ss_total != 0 else float("nan")
    )

    # Display performance metrics.
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{currency}{mae:,.2f}")
    m2.metric("RMSE", f"{currency}{rmse:,.2f}")
    m3.metric("R²", f"{r2:.4f}")

    # Plot actual and predicted prices.
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=comparison["Date"],
        y=comparison["Actual"],
        name="Actual Close",
        mode="lines",
        line=dict(color="#00D4AA", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=comparison["Date"],
        y=comparison["Predicted"],
        name="Predicted Close",
        mode="lines",
        line=dict(color="#4DA3FF", width=2),
        fill="tonexty",
        fillcolor="rgba(77, 163, 255, 0.08)",
    ))

    fig.update_layout(
        title="Actual vs Predicted — Test Set",
        template="plotly_dark",
        paper_bgcolor="#131D2E",
        plot_bgcolor="#131D2E",
        font=dict(color="#E6EDF7"),
        height=450,
        margin=dict(l=20, r=20, t=65, b=25),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            title="Trading Date",
            showgrid=False,
        ),
        yaxis=dict(
            title=f"Closing Price ({currency})",
            gridcolor="#263449",
        ),
    )

    st.plotly_chart(fig, width="stretch")
