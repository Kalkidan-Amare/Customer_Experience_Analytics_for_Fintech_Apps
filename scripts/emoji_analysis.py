import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import emoji
import pandas as pd
import matplotlib

# Headless backend for safe plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_PATH = Path("data/processed/reviews_clean.csv")
SENTIMENT_PATH = Path("data/processed/reviews_sentiment_partial.csv")
EMOJI_COUNTS_OUT = Path("data/processed/emoji_counts.csv")
EMOJI_SENTIMENT_OUT = Path("data/processed/emoji_sentiment.csv")
FIG_DIR = Path("outputs/figures")

# Regex-based fallback for emoji extraction (in case library misses edge cases)
EMOJI_PATTERN = emoji.get_emoji_regexp()


def extract_emojis(text: str) -> Iterable[str]:
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0) for m in EMOJI_PATTERN.finditer(text)]


def plot_top_emojis(df_counts: pd.DataFrame, topn: int = 10):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for bank, sub in df_counts.groupby("bank"):
        top = sub.sort_values("count", ascending=False).head(topn)
        if top.empty:
            continue
        plt.figure(figsize=(8, 5))
        # Show emoji labels on y-axis
        plt.barh(top["emoji"], top["count"], color="#4C72B0")
        plt.gca().invert_yaxis()
        plt.title(f"Top {topn} Emojis: {bank}")
        plt.xlabel("Count")
        plt.tight_layout()
        out_path = FIG_DIR / f"top_emojis_{bank.replace(' ', '_').lower()}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEAN_PATH}")
    clean_df = pd.read_csv(CLEAN_PATH)

    # Prepare per-bank emoji counts
    rows = []
    for bank, sub in clean_df.groupby("bank"):
        counter = Counter()
        for txt in sub["review"].fillna("").astype(str):
            counter.update(extract_emojis(txt))
        for e, c in counter.items():
            rows.append({"bank": bank, "emoji": e, "count": int(c)})

    counts_df = pd.DataFrame(rows)
    counts_df.sort_values(["bank", "count"], ascending=[True, False], inplace=True)
    EMOJI_COUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(EMOJI_COUNTS_OUT, index=False)

    # Optional: sentiment aggregated by emoji
    if SENTIMENT_PATH.exists():
        sent_df = pd.read_csv(SENTIMENT_PATH)
        merged = clean_df.merge(sent_df[["review", "sentiment_score"]], on="review", how="left")
        agg_map: Dict[Tuple[str, str], list] = defaultdict(list)
        for _, row in merged.iterrows():
            emjs = extract_emojis(str(row.get("review", "")))
            if not emjs:
                continue
            bank = row.get("bank", "Unknown")
            sc = row.get("sentiment_score")
            for e in emjs:
                if pd.notna(sc):
                    agg_map[(bank, e)].append(float(sc))
        sent_rows = []
        for (bank, e), vals in agg_map.items():
            if not vals:
                continue
            sent_rows.append({
                "bank": bank,
                "emoji": e,
                "n": len(vals),
                "sentiment_mean": sum(vals)/len(vals),
            })
        sent_df_out = pd.DataFrame(sent_rows)
        sent_df_out.sort_values(["bank", "n"], ascending=[True, False], inplace=True)
        sent_df_out.to_csv(EMOJI_SENTIMENT_OUT, index=False)

    # Figures
    if not counts_df.empty:
        plot_top_emojis(counts_df)

    print(f"Saved emoji counts to {EMOJI_COUNTS_OUT}")
    if SENTIMENT_PATH.exists():
        print(f"Saved emoji sentiment to {EMOJI_SENTIMENT_OUT}")
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
