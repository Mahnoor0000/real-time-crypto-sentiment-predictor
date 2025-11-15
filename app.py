import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.fetch_data import get_klines, get_latest_price
from utils.model_utils import predict_next_move
from utils.sentiment_utils import analyze_sentiment


# ---------------- CONFIG & CONSTANTS ----------------

st.set_page_config(
    page_title="Real-Time Crypto Price & Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "predictions_log.csv")

SYMBOLS = {
    "Bitcoin (BTCUSDT)": "BTCUSDT",
    "Ethereum (ETHUSDT)": "ETHUSDT",
    "BNB (BNBUSDT)": "BNBUSDT",
    "Solana (SOLUSDT)": "SOLUSDT",
}


# ----------------- SIDEBAR -----------------

st.sidebar.title("⚙ Settings")

symbol_label = st.sidebar.selectbox(
    "Select Crypto Pair",
    list(SYMBOLS.keys()),
    index=0
)
symbol = SYMBOLS[symbol_label]

interval = st.sidebar.selectbox(
    "Kline Interval",
    ["1m", "5m", "15m", "1h"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.write("This app uses **Binance public API** for real-time prices and\n"
                 "**RandomForest** for short-term direction prediction.")

st.sidebar.markdown("Made for: Real-Time Stock/Crypto + Sentiment ML Project")


# ----------------- MAIN HEADER -----------------

st.title("📈 Real-Time Crypto Price & Sentiment Predictor")
st.caption("Binance + ML model + Sentiment analysis + Logging")


# ----------------- LAYOUT -----------------

col_price, col_sent = st.columns(2)


# ----------------- PRICE & PREDICTION PANEL -----------------

with col_price:
    st.subheader(f"Live Price & Prediction — {symbol}")

    # Fetch latest klines for chart
    with st.spinner("Fetching latest market data from Binance..."):
        df = get_klines(symbol, interval=interval, limit=200)

    latest_price = df["close"].iloc[-1]
    prev_price = df["close"].iloc[-2]
    pct_change = (latest_price / prev_price - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Price", f"${latest_price:.2f}")
    c2.metric("Change (last candle)", f"{pct_change:+.2f}%")
    c3.write(f"Last update: {df['open_time'].iloc[-1]}")

    # Plot
    fig = px.line(
        df,
        x="open_time",
        y="close",
        title=f"{symbol} — Recent Price ({interval} candles)",
        labels={"open_time": "Time", "close": "Close Price"},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Prediction button
    if st.button("🔮 Predict Next Move"):
        with st.spinner("Running ML model for short-term direction..."):
            pred = predict_next_move(symbol, interval=interval)

        label = "UP" if pred["label"] == 1 else "DOWN"
        color = "🟢" if label == "UP" else "🔴"

        st.markdown(f"### Prediction: {color} **{label}**")
        st.write(f"Probability UP: **{pred['prob_up']:.2%}**")
        st.write(f"Probability DOWN: **{pred['prob_down']:.2%}**")

        # Store in session for logging
        st.session_state.last_prediction = {
            "symbol": symbol,
            "time": datetime.utcnow(),
            "price": latest_price,
            "prob_up": pred["prob_up"],
            "prob_down": pred["prob_down"],
            "label": label
        }


# ----------------- SENTIMENT PANEL -----------------

with col_sent:
    st.subheader("📣 Sentiment Analysis")

    st.write("Enter a news headline, tweet, or your own view:")

    user_text = st.text_area(
        "Sentiment Text",
        placeholder="e.g. Bitcoin surges as institutions increase holdings...",
        height=150
    )

    if st.button("🧠 Analyze Sentiment"):
        score, label = analyze_sentiment(user_text)
        st.write(f"**Sentiment label:** {label}")
        st.write(f"**Polarity score:** {score:.3f}")

        st.session_state.last_sentiment = {
            "text": user_text,
            "score": score,
            "label": label
        }

        if "last_prediction" in st.session_state:
            # Auto-log when we have both prediction & sentiment
            log_entry = {
                "timestamp": st.session_state.last_prediction["time"],
                "symbol": st.session_state.last_prediction["symbol"],
                "price": st.session_state.last_prediction["price"],
                "prob_up": st.session_state.last_prediction["prob_up"],
                "prob_down": st.session_state.last_prediction["prob_down"],
                "pred_label": st.session_state.last_prediction["label"],
                "sentiment_score": score,
                "sentiment_label": label,
                "text": user_text
            }
            log_df = pd.DataFrame([log_entry])

            if os.path.exists(LOG_FILE):
                log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
            else:
                log_df.to_csv(LOG_FILE, index=False)

            st.success("✅ Logged prediction + sentiment to logs/predictions_log.csv")


# ----------------- LOG VIEWER -----------------

st.markdown("---")
st.subheader("📜 Logged Predictions & Sentiment (Monitoring)")

if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.dataframe(log_df.tail(20))
else:
    st.info("No logs yet. Make a prediction and sentiment analysis to start logging.")
