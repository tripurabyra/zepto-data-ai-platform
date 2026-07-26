import pandas as pd
from pathlib import Path


GBP_TO_INR = 105.50


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "books_raw.csv"
CLEAN_FILE = BASE_DIR / "data" / "cleaned_data.csv"


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def clean_price(value):

    if pd.isna(value):
        return None

    try:
        # Remove currency symbol and whitespace
        cleaned = str(value).replace("£", "").strip()

        return float(cleaned)

    except (ValueError, TypeError):

        return None


def clean_rating(value):

    if pd.isna(value):
        return None

    return RATING_MAP.get(
        str(value).strip(),
        None
    )


def clean_availability(value):

    if pd.isna(value):
        return None

    value = str(value).lower()

    if "in stock" in value:
        return True

    if "out of stock" in value:
        return False

    return None


def clean_data():

    print("Loading raw data...")

    df = pd.read_csv(
        RAW_FILE
    )

    print(
        f"Raw rows: {len(df)}"
    )

    # ---------------------------
    # Clean price
    # ---------------------------

    df["price_gbp"] = df["price"].apply(
        clean_price
    )

    # ---------------------------
    # Clean rating
    # ---------------------------

    df["rating"] = df["star_rating"].apply(
        clean_rating
    )

    # ---------------------------
    # Clean availability
    # ---------------------------

    df["in_stock"] = df["availability"].apply(
        clean_availability
    )

    # ---------------------------
    # Numeric median imputation
    # ---------------------------

    price_median = df["price_gbp"].median()

    rating_median = df["rating"].median()

    df["price_gbp"] = df["price_gbp"].fillna(
        price_median
    )

    df["rating"] = df["rating"].fillna(
        round(rating_median)
    )

    # ---------------------------
    # Drop rows where required
    # text fields cannot be parsed
    # ---------------------------

    before_drop = len(df)

    df = df.dropna(
        subset=[
            "title",
            "category",
            "in_stock"
        ]
    )

    dropped_rows = (
        before_drop - len(df)
    )

    # ---------------------------
    # Convert rating to integer
    # ---------------------------

    df["rating"] = df["rating"].astype(int)

    # ---------------------------
    # Convert stock boolean
    # ---------------------------

    df["in_stock"] = df["in_stock"].astype(bool)

    # ---------------------------
    # Required fixed conversion
    # ---------------------------

    df["price_inr"] = (
        df["price_gbp"]
        * GBP_TO_INR
    )

    # ---------------------------
    # Keep final columns
    # ---------------------------

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    # ---------------------------
    # Save cleaned dataset
    # ---------------------------

    df.to_csv(
        CLEAN_FILE,
        index=False
    )

    print()
    print("Cleaning complete.")
    print(
        f"Final rows: {len(df)}"
    )

    print(
        f"Dropped rows: {dropped_rows}"
    )

    print(
        f"Price median used: {price_median}"
    )

    print(
        f"Rating median used: {rating_median}"
    )

    print(
        f"Conversion rate: "
        f"1 GBP = {GBP_TO_INR} INR"
    )

    print(
        f"Saved to: {CLEAN_FILE}"
    )

    return df


if __name__ == "__main__":
    clean_data()