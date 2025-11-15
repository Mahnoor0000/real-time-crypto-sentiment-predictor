import numpy as np
from sklearn.ensemble import RandomForestClassifier


def prepare_xy(df):
    """
    Create features + label for predicting next candle direction.
    Label: 1 = price up, 0 = price down
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["target"] = (df["return"] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    return X, y


def train_model(df):
    """Train RandomForest model on last 200 candles."""
    X, y = prepare_xy(df)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    return model


def predict_next_move(model, df):
    """Predict next candle direction."""
    last = df[["open", "high", "low", "close", "volume"]].iloc[-1:]
    pred = model.predict(last)[0]
    return "📈 UP" if pred == 1 else "📉 DOWN"
