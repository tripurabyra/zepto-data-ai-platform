import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from pathlib import Path


BASE_URL = "https://books.toscrape.com/"
START_URL = urljoin(BASE_URL, "catalogue/page-1.html")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "books_raw.csv"


def get_page(url):
    """
    Download a webpage and return BeautifulSoup object.
    """
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def scrape_book_details(book_url):
    """
    Scrape detailed information for one book.
    """

    soup = get_page(book_url)

    title = soup.find("h1").get_text(strip=True)

    price_element = soup.select_one(".price_color")
    price = price_element.get_text(strip=True) if price_element else None

    rating_element = soup.select_one(".star-rating")

    if rating_element:
        rating_classes = rating_element.get("class", [])
        star_rating = next(
            (
                item
                for item in rating_classes
                if item in ["One", "Two", "Three", "Four", "Five"]
            ),
            None
        )
    else:
        star_rating = None

    availability_element = soup.select_one(".availability")

    availability = (
        availability_element.get_text(" ", strip=True)
        if availability_element
        else None
    )

    breadcrumb = soup.select(".breadcrumb li")

    category = None

    if len(breadcrumb) >= 3:
        category = breadcrumb[-2].get_text(strip=True)

    return {
        "title": title,
        "price": price,
        "star_rating": star_rating,
        "availability": availability,
        "category": category,
    }


def scrape_books():

    all_books = []

    # First five catalogue pages
    for page_number in range(1, 6):

        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

        print(f"Scraping page {page_number}: {page_url}")

        soup = get_page(page_url)

        books = soup.select("article.product_pod")

        for book in books:

            title_element = book.select_one("h3 a")
            price_element = book.select_one(".price_color")
            rating_element = book.select_one(".star-rating")
            availability_element = book.select_one(".availability")

            if not title_element:
                continue

            title = title_element.get("title")

            price = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            rating = None

            if rating_element:

                rating_classes = rating_element.get("class", [])

                rating = next(
                    (
                        item
                        for item in rating_classes
                        if item in [
                            "One",
                            "Two",
                            "Three",
                            "Four",
                            "Five"
                        ]
                    ),
                    None
                )

            availability = (
                availability_element.get_text(
                    " ",
                    strip=True
                )
                if availability_element
                else None
            )

            relative_url = title_element.get("href")

            book_url = urljoin(
                page_url,
                relative_url
            )

            # Extract category from book detail page
            detail_data = scrape_book_details(book_url)

            all_books.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": detail_data["category"]
            })

        print(
            f"Books collected so far: {len(all_books)}"
        )

    df = pd.DataFrame(all_books)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Scraping complete.")
    print(f"Total books scraped: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    scrape_books()