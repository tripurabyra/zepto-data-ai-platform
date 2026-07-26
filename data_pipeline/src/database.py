import sqlite3
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_FILE = BASE_DIR / "data" / "cleaned_data.csv"
DATABASE_FILE = BASE_DIR / "books.db"


def create_database():

    # --------------------------------------------------
    # Check cleaned CSV exists
    # --------------------------------------------------

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {CLEAN_FILE}"
        )

    print(f"Loading cleaned data from: {CLEAN_FILE}")

    df = pd.read_csv(CLEAN_FILE)

    print(f"Rows loaded: {len(df)}")
    print("Columns:")
    print(df.columns.tolist())

    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # Clean data types before SQLite insertion
    # --------------------------------------------------

    df["title"] = df["title"].astype(str)

    df["category"] = df["category"].astype(str)

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["price_inr"] = pd.to_numeric(
        df["price_inr"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Convert rating to integer
    df["rating"] = df["rating"].round().astype(int)

    # Convert Boolean to integer for SQLite
    df["in_stock"] = df["in_stock"].astype(bool).astype(int)

    # --------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------

    df = df.dropna(
        subset=[
            "title",
            "category",
            "price_gbp",
            "price_inr",
            "rating"
        ]
    )

    print(
        f"Valid rows after cleaning: {len(df)}"
    )

    # --------------------------------------------------
    # Connect to SQLite
    # --------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    # Enable foreign keys
    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

    # --------------------------------------------------
    # Drop existing tables
    # --------------------------------------------------

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    # --------------------------------------------------
    # Create categories table
    # --------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # --------------------------------------------------
    # Create books table
    # --------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # --------------------------------------------------
    # Insert unique categories
    # --------------------------------------------------

    categories = (
        df["category"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"Categories found: {len(categories)}"
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories
            (category_name)
            VALUES (?)
            """,
            (str(category),)
        )

    # --------------------------------------------------
    # Create category ID mapping
    # --------------------------------------------------

    cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    )

    category_map = {
        category_name: category_id
        for category_id, category_name
        in cursor.fetchall()
    }

    print(
        "Category mapping created:"
    )

    print(category_map)

    # --------------------------------------------------
    # Insert books
    # --------------------------------------------------

    insert_query = """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """

    inserted_count = 0

    for _, row in df.iterrows():

        category_name = str(
            row["category"]
        )

        category_id = category_map.get(
            category_name
        )

        if category_id is None:
            print(
                f"Skipping book with unknown category: "
                f"{category_name}"
            )
            continue

        # Convert values to native Python types
        title = str(row["title"])

        price_gbp = float(
            row["price_gbp"]
        )

        price_inr = float(
            row["price_inr"]
        )

        rating = int(
            row["rating"]
        )

        in_stock = int(
            row["in_stock"]
        )

        category_id = int(
            category_id
        )

        cursor.execute(
            insert_query,
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
        )

        inserted_count += 1

    # --------------------------------------------------
    # Commit changes
    # --------------------------------------------------

    connection.commit()

    # --------------------------------------------------
    # Validate database
    # --------------------------------------------------

    cursor.execute(
        "SELECT COUNT(*) FROM books"
    )

    book_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM categories"
    )

    category_count = cursor.fetchone()[0]

    print()
    print("=" * 50)
    print("DATABASE CREATION SUCCESSFUL")
    print("=" * 50)

    print(
        f"Books inserted: {inserted_count}"
    )

    print(
        f"Books in database: {book_count}"
    )

    print(
        f"Categories in database: {category_count}"
    )

    print(
        f"Database location: {DATABASE_FILE}"
    )

    # --------------------------------------------------
    # Show sample records
    # --------------------------------------------------

    sample = pd.read_sql_query(
        """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        LIMIT 5
        """,
        connection
    )

    print()
    print("Sample database records:")
    print(sample.to_string(index=False))

    # --------------------------------------------------
    # Close database
    # --------------------------------------------------

    connection.close()


if __name__ == "__main__":
    create_database()