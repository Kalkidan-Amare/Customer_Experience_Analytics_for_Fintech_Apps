import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_concat() -> pd.DataFrame:
    files = list(RAW_DIR.glob("*_reviews.csv"))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No raw review CSVs found in data/raw")
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic validation
    required = {"review", "rating", "date", "bank", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with empty review text
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df = df[df["review"].str.len() > 0]

    # Handle missing ratings
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Normalize dates to YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Remove duplicates
    df = df.drop_duplicates(subset=["review", "rating", "date", "bank"]).reset_index(drop=True)

    # Ensure metadata
    df["bank"] = df["bank"].fillna("Unknown")
    df["source"] = df["source"].fillna("Google Play")

    return df


def main():
    df = load_and_concat()
    clean_df = clean(df)
    out_path = PROCESSED_DIR / "reviews_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset: {out_path} ({len(clean_df)} rows)")


if __name__ == "__main__":
    main()
