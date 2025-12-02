from pathlib import Path

import pandas as pd
import matplotlib

# Use a non-interactive backend to avoid Tk/Tcl errors on Windows/CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

CLEAN_PATH = Path("data/processed/reviews_clean.csv")
SENTIMENT_PATH = Path("data/processed/reviews_sentiment_partial.csv")
KEYWORDS_PATH = Path("data/processed/keywords_by_bank.csv")
FIG_DIR = Path("outputs/figures")

sns.set(style="whitegrid")


def plot_rating_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="rating", hue="bank")
    plt.title("Rating Distribution by Bank")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "rating_distribution_by_bank.png", dpi=200)
    plt.close()


def plot_sentiment_bars(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sentiment_label", hue="bank")
    plt.title("Sentiment Labels by Bank (VADER)")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "sentiment_labels_by_bank.png", dpi=200)
    plt.close()


def plot_top_keywords(kw: pd.DataFrame, topn: int = 15):
    for bank, sub in kw.groupby("bank"):
        sub = sub.sort_values("score", ascending=False).head(topn)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=sub, y="term", x="score", color="#4C72B0")
        plt.title(f"Top Keywords (TF-IDF): {bank}")
        plt.xlabel("Avg TF-IDF Score")
        plt.ylabel("Term")
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / f"top_keywords_{bank.replace(' ', '_').lower()}.png", dpi=200)
        plt.close()


def wordcloud_per_bank(df: pd.DataFrame):
    for bank, sub in df.groupby("bank"):
        text = " ".join(sub["review"].dropna().astype(str).tolist())
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / f"wordcloud_{bank.replace(' ', '_').lower()}.png", dpi=200)
        plt.close()


def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Clean dataset not found at {CLEAN_PATH}")
    df_clean = pd.read_csv(CLEAN_PATH)

    # Ratings
    plot_rating_distribution(df_clean)

    # Sentiment
    if SENTIMENT_PATH.exists():
        df_sent = pd.read_csv(SENTIMENT_PATH)
        plot_sentiment_bars(df_sent)

    # Keywords
    if KEYWORDS_PATH.exists():
        kw_df = pd.read_csv(KEYWORDS_PATH)
        plot_top_keywords(kw_df)

    # Wordclouds
    wordcloud_per_bank(df_clean)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
