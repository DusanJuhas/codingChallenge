"""A beginner‑friendly Python project designed to verify fundamental data‑analysis skills"""

import sys
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==============================
# CONFIGURATION
# ==============================

CSV_FILE = "data/original.csv"
OUTPUT_FILE = "data/cleaned_data.csv"
SUMMARY_FILE = "data/summary.csv"
ROW_COUNT = 10
SHOW_PLOTS = False


# ==============================
# DATA LOADING
# ==============================

def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{path}'")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        sys.exit(1)


# ==============================
# TASK A — EXPLORATORY ANALYSIS
# ==============================

def exploratory_analysis(df: pd.DataFrame):
    print(f"\nFirst {ROW_COUNT} rows:\n")
    print(df.head(ROW_COUNT))
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nInfo:")
    print(df.info())

    missing_counts = df.isna().sum()
    print("\nMissing values (count):")
    print(missing_counts)

    missing_pct = (missing_counts / len(df) * 100).round(2)
    print("\nMissing values (%):")
    print(missing_pct.astype(str) + " %")

    duplicated = df[df.duplicated()]
    print(f"\nDuplicated rows: {len(duplicated)}")


# ==============================
# TASK B — DATA CLEANING
# ==============================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna("")

    # Convert dates
    for col in ["review_date", "publish_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Normalize text columns
    text_columns = ["movie_title", "review_text", "reviewer"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # Convert rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    return df


# ==============================
# TASK C — STATISTICS
# ==============================

def movie_statistics(df: pd.DataFrame):

    if {"movie_title", "rating"}.issubset(df.columns):

        print("\nRating Statistics")
        print("Mean:", df["rating"].mean())
        print("Median:", df["rating"].median())
        print("Min:", df["rating"].min())
        print("Max:", df["rating"].max())

        print("\nReviews per Movie:")
        print(df["movie_title"].value_counts())

        plot_rating_histogram(df)


def plot_rating_histogram(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(df["rating"].dropna(), bins=10, edgecolor="black")
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/movie_ratings_histogram.png")
    if SHOW_PLOTS:
        plt.show()


# ==============================
# TASK D — FILTERING & GROUPING
# ==============================

def filtering_grouping(df: pd.DataFrame):

    if "rating" in df.columns:
        high = df[df["rating"] >= 8]
        print("\nReviews with Rating ≥ 8")
        print(high[["movie_title", "rating"]])

    if {"movie_title", "rating"}.issubset(df.columns):
        avg_rating = (
            df.groupby("movie_title")["rating"]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nAverage Rating per Movie")
        print(avg_rating)


# ==============================
# TASK E — VISUALIZATION
# ==============================

def plot_reviews_over_time(df: pd.DataFrame):

    if "review_date" not in df.columns:
        return

    reviews = (
        df.set_index("review_date")
        .sort_index()
        .resample("ME")
        .size()
    )

    if not reviews.empty:
        plt.figure(figsize=(8, 5))
        reviews.plot(marker="o")
        plt.title("Reviews Over Time")
        plt.tight_layout()
        plt.savefig("plots/reviews_over_time.png")
        if SHOW_PLOTS:
            plt.show()


# ==============================
# TASK F — ENHANCEMENTS
# ==============================

def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    if "review_text" in df.columns:
        df["word_count"] = df["review_text"].fillna("").apply(lambda x: len(x.split()))
    return df


def add_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if "rating" in df.columns:
        df["sentiment"] = np.where(
            df["rating"] >= 7, "positive",
            np.where(df["rating"] >= 4, "neutral", "negative")
        )
    return df


def extract_top_keywords(df: pd.DataFrame, top_n: int = 10):
    if "review_text" not in df.columns:
        return []

    stopwords = {"the", "and", "a", "to", "of", "in", "is", "it"}

    words = []
    for text in df["review_text"].fillna("").str.lower():
        words.extend([w for w in text.split() if w not in stopwords])

    return Counter(words).most_common(top_n)


# ==============================
# TASK G — EXPORT
# ==============================

def export_results(df: pd.DataFrame):

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_FILE}")

    summary = {
        "count_reviews": len(df),
        "rating_mean": df["rating"].mean() if "rating" in df else None,
        "unique_movies": df["movie_title"].nunique() if "movie_title" in df else None
    }

    pd.DataFrame(summary.items(), columns=["metric", "value"]) \
        .to_csv(SUMMARY_FILE, index=False)

    print(f"Saved summary to {SUMMARY_FILE}")


# ==============================
# MAIN EXECUTION
# ==============================

def main():
    df = load_dataset(CSV_FILE)

    exploratory_analysis(df)

    df = clean_data(df)

    movie_statistics(df)

    filtering_grouping(df)

    plot_reviews_over_time(df)

    df = add_word_count(df)
    df = add_sentiment_proxy(df)

    keywords = extract_top_keywords(df)
    print("\nTop Keywords:", keywords)

    export_results(df)


if __name__ == "__main__":
    main()