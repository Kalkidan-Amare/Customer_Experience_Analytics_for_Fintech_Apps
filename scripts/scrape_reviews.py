import argparse
import csv
from datetime import datetime
from typing import List, Dict

from google_play_scraper import Sort, reviews

BANK_APPS = {
    
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",  
    "Bank of Abyssinia": "com.infonow.bofa",     
    "Dashen Bank": "com.dashen.dashensuperapp"     
}


def fetch_reviews(app_id: str, lang: str = "en", country: str = "us", count: int = 500) -> List[Dict]:
    all_reviews: List[Dict] = []
    next_token = None

    while len(all_reviews) < count:
        batch, next_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=min(200, count - len(all_reviews)),
            continuation_token=next_token,
        )
        all_reviews.extend(batch)
        if not next_token:
            break
    return all_reviews[:count]


def save_csv(rows: List[Dict], bank_name: str, output_path: str) -> None:
    fieldnames = ["review", "rating", "date", "bank", "source"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "review": r.get("content", "").strip(),
                "rating": r.get("score", None),
                "date": r.get("at").strftime("%Y-%m-%d") if isinstance(r.get("at"), datetime) else "",
                "bank": bank_name,
                "source": "Google Play"
            })


def main():
    parser = argparse.ArgumentParser(description="Scrape Google Play reviews for Ethiopian bank apps.")
    parser.add_argument("--per_bank", type=int, default=500, help="Number of reviews per bank to fetch")
    parser.add_argument("--lang", type=str, default="en", help="Language code")
    parser.add_argument("--country", type=str, default="us", help="Country code")
    args = parser.parse_args()

    for bank, app_id in BANK_APPS.items():
        print(f"Fetching reviews for {bank} ({app_id})...")
        rows = fetch_reviews(app_id, lang=args.lang, country=args.country, count=args.per_bank)
        output_path = f"data/raw/{bank.replace(' ', '_').lower()}_reviews.csv"
        save_csv(rows, bank, output_path)
        print(f"Saved {len(rows)} reviews to {output_path}")


if __name__ == "__main__":
    main()
