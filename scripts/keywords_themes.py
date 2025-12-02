import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_PATH = Path("data/processed/reviews_clean.csv")
SENTIMENT_PATH = Path("data/processed/reviews_sentiment_partial.csv")
THEMES_OUT = Path("data/processed/reviews_themes.csv")
KEYWORDS_OUT = Path("data/processed/keywords_by_bank.csv")

# Simple, rule-based theme definitions. Adjust as needed.
THEME_RULES: Dict[str, List[str]] = {
    "Transaction Performance": [
        r"\bslow\b", r"loading", r"lag", r"delay", r"timeout", r"transfer[s]?", r"processing",
    ],
    "Reliability & Access": [
        r"crash", r"error", r"bug", r"fail", r"freeze", r"login", r"verify", r"otp", r"session",
    ],
    "UX & Navigation": [
        r"ui", r"ux", r"interface", r"design", r"navigation", r"confus", r"hard to", r"layout",
    ],
    "Features & Requests": [
        r"feature", r"fingerprint|biometric|face id", r"notification", r"statement", r"beneficiar", r"template",
    ],
    "Support & Service": [
        r"support", r"help", r"response", r"call", r"email", r"contact",
    ],
}


def _top_tfidf_keywords(df: pd.DataFrame, text_col: str, n: int = 30) -> List[Tuple[str, float]]:
    corpus = df[text_col].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(corpus)
    # Average TF-IDF across documents
    avg_scores = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = list(zip(terms, avg_scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def _assign_themes(text: str) -> List[str]:
    text_l = text.lower()
    matched = []
    for theme, patterns in THEME_RULES.items():
        for pat in patterns:
            if re.search(pat, text_l):
                matched.append(theme)
                break
    return matched if matched else ["Other"]


def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)

    # Generate per-bank top keywords using TF-IDF
    banks = sorted(df["bank"].dropna().unique())
    kw_rows = []
    for bank in banks:
        sub = df[df["bank"] == bank]
        if len(sub) == 0:
            continue
        top_kw = _top_tfidf_keywords(sub, "review", n=30)
        for rank, (term, score) in enumerate(top_kw, start=1):
            kw_rows.append({"bank": bank, "rank": rank, "term": term, "score": float(score)})
    kw_df = pd.DataFrame(kw_rows)
    KEYWORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    kw_df.to_csv(KEYWORDS_OUT, index=False)

    # Assign themes to each review via keyword rules
    df["themes"] = df["review"].fillna("").astype(str).apply(lambda t: ", ".join(_assign_themes(t)))

    THEMES_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(THEMES_OUT, index=False)
    print(f"Saved themes per review to {THEMES_OUT} and keywords to {KEYWORDS_OUT}")


if __name__ == "__main__":
    main()
