# 📈 Real-Time Crypto Price & Sentiment Predictor

A modern Streamlit-based dashboard that combines **real-time crypto market data**, **ML-based price prediction**, and **sentiment analysis**. This project is built as an industry-style financial analytics tool, demonstrating real-time data ingestion, model inference, visualization, and NLP insights.

---

## 🚀 What This Project Does

### 🔹 Live Cryptocurrency Tracking
Fetches live market data for BTC, ETH, BNB, and SOL using the Binance API.  
Supports multiple time intervals such as **1m, 5m, 15m, 1h, 4h**.  
Displays:
- Latest price  
- Percentage change (last candle)  
- Last update timestamp  
- Interactive price chart (Plotly)

---

### 🔹 Machine Learning Price Prediction
Uses a lightweight **Random Forest model** trained on recent candlestick data.  
Predicts the expected **next movement** (UP or DOWN) in real-time.  
Fast inference suitable for live dashboards.

---

### 🔹 Sentiment Analysis (NLP)
Allows you to analyze any text (news headline, tweet, market analysis).  
The system returns:
- **Sentiment label** — Positive / Neutral / Negative  
- **Polarity score** using NLP techniques  

---

### 🔹 Clean & Responsive UI
The dashboard includes:
- A dark-themed modern interface  
- Sidebar controls  
- Metric cards  
- Smooth price chart  
- Organized layout for clarity  

---

## 🧠 Technologies Used
- **Streamlit** — UI & app framework  
- **Binance API** — real-time crypto data  
- **Random Forest** — ML classifier  
- **TextBlob** — sentiment scoring  
- **Plotly** — interactive charts  

---

