import pandas as pd
from scripts.preprocess_reviews import clean


def test_clean_basic():
    df = pd.DataFrame([
        {"review": "Great app", "rating": 5, "date": "2025-11-28", "bank": "CBE", "source": "Google Play"},
        {"review": "Great app", "rating": 5, "date": "2025-11-28", "bank": "CBE", "source": "Google Play"},  # duplicate
        {"review": "", "rating": 4, "date": "2025-11-29", "bank": "BOA", "source": "Google Play"},  # empty
        {"review": "slow loading", "rating": 2, "date": "2025-11-29", "bank": "Dashen", "source": "Google Play"},
    ])

    out = clean(df)
    # One duplicate removed, one empty removed -> expect 2 rows
    assert len(out) == 2
    assert set(out["bank"]) == {"CBE", "Dashen"}
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(out["date"], errors="coerce"))
