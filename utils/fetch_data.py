import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"
PRICE_URL = "https://api.binance.com/api/v3/ticker/price"


def get_klines(symbol="BTCUSDT", interval="1m", limit=200):
    """Fetch OHLCV candles from Binance and return a clean DataFrame."""
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        raise RuntimeError("Invalid Binance response:", data)

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
        ],
    )

    # Convert numeric columns
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()

    return df


def get_latest_price(symbol="BTCUSDT"):
    """Returns latest price of a symbol from Binance."""
    r = requests.get(PRICE_URL, params={"symbol": symbol})
    r.raise_for_status()
    return float(r.json()["price"])
