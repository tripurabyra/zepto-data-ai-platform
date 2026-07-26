import sqlite3
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_FILE = BASE_DIR / "books.db"

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "query_results"
)

COMPARISON_DIR = (
    BASE_DIR
    / "output"
    / "comparison_results"
)


def run_queries():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    queries = {

        "query_1_select_where": """
            SELECT
                title,
                price_gbp,
                rating,
                in_stock
            FROM books
            WHERE in_stock = 1
            AND rating >= 4
        """,

        "query_2_order_by": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            ORDER BY price_gbp DESC
        """,

        "query_3_limit": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            ORDER BY rating DESC
            LIMIT 10
        """,

        "query_4_distinct": """
            SELECT DISTINCT
                category_name
            FROM categories
            ORDER BY category_name
        """,

        "query_5_between": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp
        """,

        "query_6_join": """
            SELECT
                c.category_name,
                b.title,
                b.price_gbp,
                b.price_inr,
                b.rating,
                b.in_stock
            FROM books b
            INNER JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY
                c.category_name,
                b.rating DESC,
                b.title
        """
    }

    query_outputs = {}

    # ---------------------------
    # Execute SQL queries
    # ---------------------------

    for query_name, query in queries.items():

        print()
        print("=" * 60)
        print(query_name)
        print("=" * 60)

        print(query)

        result = pd.read_sql(
            query,
            connection
        )

        print(result)

        output_file = (
            OUTPUT_DIR
            / f"{query_name}.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        query_outputs[
            query_name
        ] = result

    # ---------------------------
    # Load tables into pandas
    # ---------------------------

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    # ---------------------------
    # Reproduce JOIN using pandas
    # ---------------------------

    pandas_join = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_join = pandas_join[
        [
            "category_name",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock"
        ]
    ]

    pandas_join = pandas_join.sort_values(
        by=[
            "category_name",
            "rating",
            "title"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )

    pandas_join = pandas_join.reset_index(
        drop=True
    )

    sql_join = query_outputs[
        "query_6_join"
    ].copy()

    sql_join = sql_join.reset_index(
        drop=True
    )

    # ---------------------------
    # Compare SQL and pandas
    # ---------------------------

    equivalent = pandas_join.equals(
        sql_join
    )

    print()
    print("=" * 60)
    print("SQL JOIN vs pandas.merge()")
    print("=" * 60)

    print(
        f"Equivalent: {equivalent}"
    )

    pandas_join.to_csv(
        COMPARISON_DIR
        / "pandas_merge_result.csv",
        index=False
    )

    sql_join.to_csv(
        COMPARISON_DIR
        / "sql_join_result.csv",
        index=False
    )

    comparison_summary = pd.DataFrame(
        {
            "check": [
                "SQL JOIN row count",
                "pandas merge row count",
                "Results equivalent"
            ],
            "value": [
                len(sql_join),
                len(pandas_join),
                equivalent
            ]
        }
    )

    comparison_summary.to_csv(
        COMPARISON_DIR
        / "comparison_summary.csv",
        index=False
    )

    connection.close()


if __name__ == "__main__":
    run_queries()