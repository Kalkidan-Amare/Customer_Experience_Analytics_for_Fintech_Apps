import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_PATH = Path("data/processed/reviews_clean.csv")
OUTPUT_PATH = Path("data/processed/reviews_sentiment_partial.csv")


def label_from_score(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Expected cleaned reviews at {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    analyzer = SentimentIntensityAnalyzer()

    scores = df["review"].fillna("").astype(str).apply(analyzer.polarity_scores)
    df["sentiment_score"] = scores.apply(lambda s: s["compound"])  # -1..1
    df["sentiment_label"] = df["sentiment_score"].apply(label_from_score)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved partial sentiment results: {OUTPUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
