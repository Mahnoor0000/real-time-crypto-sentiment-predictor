import streamlit as st
import plotly.express as px
from utils.fetch_data import get_klines, get_latest_price
from utils.model_utils import train_model, predict_next_move
from utils.sentiment_utils import analyze_sentiment

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Crypto Predictor",
    page_icon="📈",
    layout="wide"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.metric-card {
    padding: 18px 22px;
    background: #161A23;
    border: 1px solid #2C2F38;
    border-radius: 12px;
    text-align: center;
}

.metric-label {
    font-size: 15px;
    color: #B0B0B0;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: white;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

.sentiment-box textarea {
    background-color: #1B1E27 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #333 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.title("📈 Real-Time Crypto Price & Sentiment Predictor")

# ---------------------- SIDEBAR ----------------------
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

# ---------------------- FETCH DATA ----------------------
st.markdown(f"<div class='section-title'>Live Price & Prediction — <b>{symbol_code}</b></div>", unsafe_allow_html=True)

df = get_klines(symbol_code, interval=interval, limit=200)
latest_price = get_latest_price(symbol_code)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest Price</div>
        <div class="metric-value">${latest_price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

pct = df["close"].pct_change().iloc[-1] * 100

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Change (last candle)</div>
        <div class="metric-value">{pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Update</div>
        <div class="metric-value">{str(df["time"].iloc[-1])}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- PLOT CHART ----------------------
fig = px.line(
    df, x="time", y="close",
    title="",
)
fig.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=10, r=10, t=20, b=10)
)

st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- ML PREDICTION ----------------------
st.markdown("<div class='section-title'>🤖 ML Price Direction Prediction</div>", unsafe_allow_html=True)

model = train_model(df)
prediction = predict_next_move(model, df)

st.success(f"Next move: **{prediction}**")

# ---------------------- SENTIMENT ----------------------
st.markdown("<div class='section-title'>🧠 Sentiment Analysis</div>", unsafe_allow_html=True)

text = st.text_area(
    "Enter news, tweet, or your analysis:",
    key="sentiment_area",
    height=140
)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please type something.")
    else:
        polarity, label = analyze_sentiment(text)
        st.info(f"**Sentiment:** {label}")
        st.write(f"**Polarity Score:** `{polarity:.4f}`")
