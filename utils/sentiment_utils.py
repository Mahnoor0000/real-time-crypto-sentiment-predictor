from textblob import TextBlob


def analyze_sentiment(text: str):
    """
    Simple sentiment analysis using TextBlob polarity.
    Returns score (-1 to 1) and label: negative/neutral/positive
    """
    if not text or text.strip() == "":
        return 0.0, "neutral"

    blob = TextBlob(text)
    score = blob.sentiment.polarity  # -1..1

    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return score, label
