import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (Column, Date, Float, ForeignKey, Integer, MetaData,
                        String, Table, create_engine, text)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

CLEAN_PATH = Path("data/processed/reviews_clean.csv")
SENTIMENT_PATH = Path("data/processed/reviews_sentiment_partial.csv")


def get_engine_from_env() -> Engine:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "root")
    db = os.getenv("PGDATABASE", "bank_reviews")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, echo=False, future=True)


def define_schema(metadata: MetaData) -> tuple[Table, Table]:
    banks = Table(
        "banks",
        metadata,
        Column("bank_id", Integer, primary_key=True, autoincrement=True),
        Column("bank_name", String(255), nullable=False, unique=True),
        Column("app_name", String(255), nullable=True),
    )

    reviews = Table(
        "reviews",
        metadata,
        Column("review_id", Integer, primary_key=True, autoincrement=True),
        Column("bank_id", Integer, ForeignKey("banks.bank_id"), nullable=False),
        Column("review_text", String, nullable=False),
        Column("rating", Integer, nullable=True),
        Column("review_date", Date, nullable=True),
        Column("sentiment_label", String(32), nullable=True),
        Column("sentiment_score", Float, nullable=True),
        Column("source", String(64), nullable=True),
    )
    return banks, reviews


def upsert_banks(conn, banks_table: Table, df: pd.DataFrame) -> dict:
    bank_name_to_id: dict[str, int] = {}
    existing = conn.execute(text("SELECT bank_id, bank_name FROM banks"))
    for row in existing:
        bank_name_to_id[row.bank_name] = row.bank_id

    unique_banks = sorted(set(df["bank"].dropna().tolist()))
    for name in unique_banks:
        if name not in bank_name_to_id:
            res = conn.execute(banks_table.insert().values(bank_name=name, app_name=None).returning(banks_table.c.bank_id))
            bank_id = res.scalar_one()
            bank_name_to_id[name] = bank_id
    return bank_name_to_id


def load_reviews(conn, reviews_table: Table, bank_name_to_id: dict, df: pd.DataFrame):
    records = []
    for _, r in df.iterrows():
        records.append({
            "bank_id": bank_name_to_id.get(r.get("bank")),
            "review_text": str(r.get("review", ""))[:10000],
            "rating": int(r["rating"]) if pd.notna(r.get("rating")) else None,
            "review_date": pd.to_datetime(r.get("date"), errors="coerce").date() if pd.notna(r.get("date")) else None,
            "sentiment_label": r.get("sentiment_label"),
            "sentiment_score": float(r.get("sentiment_score")) if pd.notna(r.get("sentiment_score")) else None,
            "source": r.get("source"),
        })
    if records:
        conn.execute(reviews_table.insert(), records)


def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Expected cleaned dataset at {CLEAN_PATH}")

    df_clean = pd.read_csv(CLEAN_PATH)
    if SENTIMENT_PATH.exists():
        df_sent = pd.read_csv(SENTIMENT_PATH)
        # Merge to include sentiment columns if available
        merge_cols = [c for c in df_clean.columns if c != "review"] + ["review"]
        df = df_clean.merge(df_sent[["review", "sentiment_label", "sentiment_score"]], on="review", how="left")
    else:
        df = df_clean.copy()
        df["sentiment_label"] = None
        df["sentiment_score"] = None

    engine = get_engine_from_env()
    metadata = MetaData()
    banks_t, reviews_t = define_schema(metadata)

    try:
        with engine.begin() as conn:
            metadata.create_all(conn)
            bank_map = upsert_banks(conn, banks_t, df)
            load_reviews(conn, reviews_t, bank_map, df)

            # Example checks
            print("Counts per bank:")
            res = conn.execute(text("SELECT b.bank_name, COUNT(*) AS cnt FROM reviews r JOIN banks b ON r.bank_id=b.bank_id GROUP BY b.bank_name ORDER BY cnt DESC"))
            for row in res:
                print(f"  {row.bank_name}: {row.cnt}")

            print("Average rating per bank:")
            res2 = conn.execute(text("SELECT b.bank_name, AVG(r.rating)::numeric(10,2) AS avg_rating FROM reviews r JOIN banks b ON r.bank_id=b.bank_id GROUP BY b.bank_name ORDER BY avg_rating DESC"))
            for row in res2:
                print(f"  {row.bank_name}: {row.avg_rating}")

        print("Database load completed.")
    except SQLAlchemyError as e:
        print(f"Database error: {e}")


if __name__ == "__main__":
    main()
