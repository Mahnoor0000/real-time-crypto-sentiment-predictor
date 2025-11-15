import requests
import pandas as pd
from datetime import datetime


BINANCE_BASE = "https://api.binance.com"


def get_klines(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical klines (candles) from Binance.
    symbol: e.g. 'BTCUSDT'
    interval: e.g. '1m', '5m'
    limit: up to 1000
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # Kline format:
    # [
    #   [
    #     1499040000000,      // Open time
    #     "0.01634790",       // Open
    #     "0.80000000",       // High
    #     "0.01575800",       // Low
    #     "0.01577100",       // Close
    #     "148976.11427815",  // Volume
    #     1499644799999,      // Close time
    #     ...
    #   ]
    # ]

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    # Convert types
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)

    df["open_time"] = df["open_time"].apply(
        lambda x: datetime.fromtimestamp(x / 1000.0)
    )
    df["close_time"] = df["close_time"].apply(
        lambda x: datetime.fromtimestamp(x / 1000.0)
    )

    return df[["open_time", "open", "high", "low", "close", "volume"]]


def get_latest_price(symbol: str) -> float:
    """
    Fetch latest price using /ticker/price endpoint.
    """
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    resp = requests.get(url, params={"symbol": symbol})
    resp.raise_for_status()
    data = resp.json()
    return float(data["price"])
