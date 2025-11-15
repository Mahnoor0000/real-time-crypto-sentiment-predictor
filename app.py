import streamlit as st
import plotly.express as px
from utils.fetch_data import get_klines, get_latest_price
from utils.model_utils import train_model, predict_next_move
from utils.sentiment_utils import analyze_sentiment

# ---- UI CONFIG ----
st.set_page_config(
    page_title="Real-Time Crypto Price & Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Real-Time Crypto Price & Sentiment Predictor")

# Sidebar
st.sidebar.header("⚙ Settings")

pairs = {
    "Bitcoin (BTCUSDT)": "BTCUSDT",
    "Ethereum (ETHUSDT)": "ETHUSDT",
    "BNB (BNBUSDT)": "BNBUSDT",
    "Solana (SOLUSDT)": "SOLUSDT"
}
intervals = ["1m", "5m", "15m", "1h", "4h"]

symbol = st.sidebar.selectbox("Select Pair", list(pairs.keys()))
interval = st.sidebar.selectbox("Kline Interval", intervals)
symbol_code = pairs[symbol]


# ---- Fetch Data ----
st.subheader(f"Live Price & Prediction — {symbol_code}")

df = get_klines(symbol_code, interval=interval, limit=200)
latest_price = get_latest_price(symbol_code)

c1, c2, c3 = st.columns(3)
c1.metric("Latest Price", f"${latest_price:,.2f}")
pct = df["close"].pct_change().iloc[-1] * 100
c2.metric("Change (last candle)", f"{pct:.2f}%")
c3.metric("Last Update", str(df["time"].iloc[-1]))


# ---- Plot Chart ----
fig = px.line(df, x="time", y="close", title=f"{symbol_code} — Recent Price")
st.plotly_chart(fig, use_container_width=True)


# ---- ML Prediction ----
model = train_model(df)
prediction = predict_next_move(model, df)

st.subheader("🤖 ML Price Direction Prediction")
st.success(f"Next move: **{prediction}**")


# ---- Sentiment ----
st.subheader("🧠 Sentiment Analysis")

text = st.text_area("Enter news, tweet, or your analysis:")
if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please type something.")
    else:
        polarity, label = analyze_sentiment(text)
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Polarity Score:** {polarity:.4f}")
