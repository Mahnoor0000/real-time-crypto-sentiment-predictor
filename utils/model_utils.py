import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .fetch_data import get_klines


MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for ML model from OHLCV data.
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ret_1"] = df["return"].shift(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["vol_5"] = df["return"].rolling(5).std()
    df["vol_10"] = df["return"].rolling(10).std()

    df["future_ret"] = df["close"].shift(-1) / df["close"] - 1.0
    df["target"] = (df["future_ret"] > 0).astype(int)

    df = df.dropna()

    features = [
        "ret_1", "ret_3", "ret_5",
        "ma_5", "ma_10",
        "vol_5", "vol_10"
    ]
    X = df[features]
    y = df["target"].astype(int)
    return X, y, features


def train_model_for_symbol(symbol: str, interval: str = "1m") -> Tuple[RandomForestClassifier, StandardScaler, list]:
    """
    Fetch data, engineer features, train a RandomForest model.
    Returns model, scaler, feature list.
    """
    df = get_klines(symbol, interval=interval, limit=500)
    X, y, features = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    features_path = os.path.join(MODEL_DIR, f"{symbol}_features.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "wb") as f:
        pickle.dump(features, f)

    return model, scaler, features


def load_model_for_symbol(symbol: str):
    """
    Load model, scaler, and features; train if not existing.
    """
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    features_path = os.path.join(MODEL_DIR, f"{symbol}_features.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        return train_model_for_symbol(symbol)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        features = pickle.load(f)

    return model, scaler, features


def prepare_latest_features(symbol: str, interval: str = "1m") -> np.ndarray:
    """
    Get most recent data and build a single feature row for prediction.
    """
    df = get_klines(symbol, interval=interval, limit=60)
    X, _, features = engineer_features(df)
    latest_row = X.iloc[-1]  # last row
    return latest_row.values.reshape(1, -1), features


def predict_next_move(symbol: str, interval: str = "1m"):
    """
    High-level helper:
    - load model (train if missing)
    - fetch latest features
    - return predicted probability of up move & label
    """
    model, scaler, features = load_model_for_symbol(symbol)
    X_latest, feat_latest = prepare_latest_features(symbol, interval=interval)

    # ensure feature order matches
    assert list(feat_latest) == list(features), "Feature mismatch"

    X_scaled = scaler.transform(X_latest)
    proba = model.predict_proba(X_scaled)[0]
    pred_label = int(model.predict(X_scaled)[0])  # 1 = up, 0 = down

    return {
        "prob_down": float(proba[0]),
        "prob_up": float(proba[1]),
        "label": pred_label
    }
