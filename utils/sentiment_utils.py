from textblob import TextBlob


def analyze_sentiment(text: str):
    """Return sentiment polarity and classification."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        label = "😊 Positive"
    elif polarity < -0.1:
        label = "😟 Negative"
    else:
        label = "😐 Neutral"

    return polarity, label
