"""A beginner‑friendly Python project designed to verify fundamental data‑analysis skills"""

import sys
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# csv_file is the path to the CSV file containing the film reviews dataset
CVS_FILE = "data/original.csv"

# row_count is the number of rows to display when showing the first few rows of the dataset
ROW_COUNT = 10

# Set to False to skip showing plots (useful for non-interactive environments)
SHOW_PLOTS = False

# ============================================
#        TASK A — EXPLORATORY DATA ANALYSIS
# ============================================

try:
# task A.1 → Load the dataset
    df = pd.read_csv(CVS_FILE)
    print(f"First {ROW_COUNT} rows of the dataset:\n")
# task A.2.a → Display first 10 rows
    print(df.head(ROW_COUNT))
# task A.2.b → Display the shape of the dataset
    print(df.shape)
# task A.2.c → Display the column names
    print(df.columns)
# task A.2.d → Display the data types by using the info() method
    print(df.info())

# task A.3.a → Display Missing values
    print("\nMissing values per column (count):")
    missing_counts = df.isna().sum()
    print(missing_counts)

    print("\nMissing values per column (percentage):")
    missing_pct = (missing_counts / len(df) * 100).round(2)
    print(missing_pct.astype(str) + " %")

# Rows that have ANY missing values
    rows_with_missing = df[df.isna().any(axis=1)]
    print(f"\nTotal rows with any missing value: {len(rows_with_missing)}")

    if len(rows_with_missing) > 0:
        print(f"\nShowing up to the rows with missing values: \n{rows_with_missing}")

# task A.3.b → Display Duplicated values
    duplicated_rows = df[df.duplicated()]
    print(f"\nTotal duplicated rows: {len(duplicated_rows)}")
    if len(duplicated_rows) > 0:
        print(f"\nShowing up to the duplicated rows: \n{duplicated_rows}")

except FileNotFoundError:
    print(f"Error: File not found at path '{CVS_FILE}'.")
    sys.exit(1)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    sys.exit(1)

# ================================
#        TASK B — DATA CLEANING
# ================================

# task B.1 → remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n=== Shape after duplicate removal: {df.shape} ===")

# task B.2 → handle missing values by filling with empty string
# empty_string is a variable to be used to fill missing values in the dataset
EMPTY_STRING = ""
df = df.fillna(EMPTY_STRING)
print(f"\nMissing values FILLED with: '{EMPTY_STRING}'")

# task B.3 → convert date columns to datetime
# datafield name for date in the dataset
DATE_COLUMN = "review_date"
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
print(f"Converted '{DATE_COLUMN}' to datetime.")

# task B.4 → Trim whitespace in text fields
# a list of column names that contain text data and may require trimming and normalization
text_columns = ["movie_title", "review_text", "reviewer"]
for col in text_columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].str.lower() # Normalize text to lowercase
    print(f"Text normalized in column '{col}'.")

# task B.5 → Convert numeric columns to proper dtype
numeric_columns = ["rating"]  # Example numeric column
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Converted '{col}' to numeric dtype.")

# ================================
#        TASK C — STATISTICS
# ================================

print("\n=== TASK C: BASIC STATISTICAL ANALYSIS ===")

# -------------------------------------------
# C.1 — MOVIE REVIEWS STATISTICS
# -------------------------------------------

print("\n--- C.1 MOVIE REVIEW STATISTICS ---")

if "rating" in df.columns:

    # Mean, median, min, max rating
    print("\nRating Statistics:")
    print(f"Mean rating:  {df['rating'].mean():.2f}")
    print(f"Median rating:{df['rating'].median():.2f}")
    print(f"Min rating:   {df['rating'].min()}")
    print(f"Max rating:   {df['rating'].max()}")

    # Count of reviews per movie
    print("\nReviews per Movie:")
    review_counts = df["movie_title"].value_counts()
    print(review_counts)

    # Distribution of ratings - histogram

    plt.figure(figsize=(8, 5))
    plt.hist(df["rating"], bins=10, edgecolor="black")
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/movie_ratings_histogram.png")
    if SHOW_PLOTS:
        plt.show()

else:
    print("No movie review fields found.")


# -------------------------------------------
# C.2 — NEWS STATISTICS (conditional)
# -------------------------------------------

print("\n--- C.2 NEWS STATISTICS (IF AVAILABLE) ---")

has_news = any(col in df.columns for col in ["category", "reviewer", "publish_date", "views"])

if not has_news:
    print("No news-related fields found in dataset.")
else:
    # Count of articles per category
    if "category" in df.columns:
        print("\nArticles per Category:")
        print(df["category"].value_counts())

    # Articles per reviewer
    if "reviewer" in df.columns:
        print("\nArticles per Reviewer:")
        print(df["reviewer"].value_counts())

    # Most frequent publish day
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        df["publish_day"] = df["publish_date"].dt.day_name()
        print("\nMost Frequent Publish Day:")
        print(df["publish_day"].value_counts().head(1))

    # Summary stats for numeric columns
    if "views" in df.columns:
        print("\nSummary Statistics for Views:")
        print(df["views"].describe())

# ================================
#        TASK D — FILTERING & GROUPING
# ================================

print("\n=== TASK D: FILTERING & GROUPING ===")

# -------------------------------------------
# D.1 — MOVIE REVIEW FILTERING & GROUPING
# -------------------------------------------

print("\n--- D.1 MOVIE REVIEWS ---")

# 1. Show reviews with rating ≥ 8
if "rating" in df.columns:
    high_ratings = df[df["rating"] >= 8]
    print("\nReviews with Rating ≥ 8:")
    print(high_ratings[["movie_title", "rating", "review_text", "reviewer"]])
else:
    print("Rating column not found — skipping this section.")

# 2. Top 5 most-reviewed movies
if "movie_title" in df.columns:
    print("\nTop 5 Most-Reviewed Movies:")
    top_movies = df["movie_title"].value_counts().head(5)
    print(top_movies)
else:
    print("movie_title column not found — skipping.")

# 3. Average rating per movie
if "rating" in df.columns and "movie_title" in df.columns:
    print("\nAverage Rating per Movie:")
    avg_rating = df.groupby("movie_title")["rating"].mean().sort_values(ascending=False)
    print(avg_rating)
else:
    print("Missing columns for average rating calculation.")


# -------------------------------------------
# D.2 — NEWS FILTERING & GROUPING
# -------------------------------------------

print("\n--- D.2 NEWS ARTICLES ---")

# Check if news data is present
news_available = any(col in df.columns for col in ["category", "author", "publish_date"])

if not news_available:
    print("No news-related fields available — skipping news analysis.")
else:
    # 1. Articles in the “Technology” category
    if "category" in df.columns:
        tech_articles = df[df["category"].str.lower() == "tech"]
        print("\nArticles in the 'Technology' Category:")
        print(tech_articles[["author", "publish_date", "views"]])
    else:
        print("category column missing — cannot filter by category.")

    # 2. Most active author
    if "author" in df.columns:
        print("\nMost Active Author:")
        most_active = df["author"].value_counts().head(1)
        print(most_active)
    else:
        print("author column missing — cannot compute most active author.")

    # 3. Number of articles published each month
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        df["month"] = df["publish_date"].dt.month_name()

        print("\nArticles Published Per Month:")
        month_counts = df["month"].value_counts()
        print(month_counts)
    else:
        print("publish_date column missing — cannot compute monthly article output.")

# ================================
#        TASK E — DATA VISUALIZATION (Matplotlib)
# ================================
print("\n=== TASK E: DATA VISUALIZATION (Matplotlib) ===")

# Ensure dates are datetime if present
if "review_date" in df.columns:
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
if "publish_date" in df.columns:
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

# -------------------------------------------
# E.1 — MOVIE REVIEWS
# -------------------------------------------
print("\n--- E.1 MOVIE REVIEWS ---")

# 1) Histogram: rating distribution
if "rating" in df.columns:
    plt.figure(figsize=(8, 5))
    # Use integer bins 0..10 (inclusive edges)
    bins = range(int(df["rating"].min()) if pd.notna(df["rating"].min()) else 0,
                 int(df["rating"].max()) + 2 if pd.notna(df["rating"].max()) else 11)
    plt.hist(df["rating"].dropna(), bins=bins, edgecolor="black")
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks(range(min(bins), max(bins)))
    plt.tight_layout()
    plt.savefig("plots/movie_ratings_histogram.png")
    if SHOW_PLOTS:
        plt.show()
else:
    print("Skipping rating histogram — 'rating' column not found.")

# 2) Bar plot: average rating per movie
if {"movie_title", "rating"}.issubset(df.columns):
    avg_per_movie = (
        df.groupby("movie_title")["rating"]
          .mean()
          .sort_values(ascending=False)
    )

    # Plot top N for readability
    TOP_N = 15
    to_plot = avg_per_movie.head(TOP_N)

    plt.figure(figsize=(10, 6))
    to_plot.plot(kind="bar", color="#1f77b4", edgecolor="black")
    plt.title(f"Average Rating per Movie (Top {min(TOP_N, len(to_plot))})")
    plt.xlabel("Movie")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/movie_average_ratings.png")
    if SHOW_PLOTS:
        plt.show()
else:
    print("Skipping bar plot — need both 'movie_title' and 'rating' columns.")

# 3) Line plot: number of reviews over time (monthly)
if "review_date" in df.columns:
    reviews_over_time = (
        df.set_index("review_date")
          .sort_index()
          .resample("ME")
          .size()
    )

    if not reviews_over_time.empty:
        plt.figure(figsize=(9, 5))
        reviews_over_time.plot(marker="o")
        plt.title("Number of Reviews Over Time (Monthly)")
        plt.xlabel("Month")
        plt.ylabel("Review Count")
        plt.tight_layout()
        plt.savefig("plots/reviews_over_time.png")
        if SHOW_PLOTS:
            plt.show()
    else:
        print("No valid dates in 'review_date' to plot reviews over time.")
else:
    print("Skipping reviews-over-time plot — 'review_date' not found.")


# -------------------------------------------
# E.2 — NEWS
# -------------------------------------------
print("\n--- E.2 NEWS ---")

# 1) Bar chart: articles per category
if "category" in df.columns:
    cat_counts = df["category"].dropna().astype(str).str.strip().value_counts()

    if not cat_counts.empty:
        plt.figure(figsize=(8, 5))
        cat_counts.plot(kind="bar", color="#2ca02c", edgecolor="black")
        plt.title("Articles per Category")
        plt.xlabel("Category")
        plt.ylabel("Article Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("plots/articles_per_category.png")
        if SHOW_PLOTS:
            plt.show()
    else:
        print("No non-empty values in 'category' to plot.")
else:
    print("Skipping category bar chart — 'category' column not found.")

# 2) Line plot: articles per month
if "publish_date" in df.columns:
    articles_per_month = (
        df.set_index("publish_date")
          .sort_index()
          .resample("ME")
          .size()
    )

    if not articles_per_month.empty:
        plt.figure(figsize=(9, 5))
        articles_per_month.plot(marker="o", color="#ff7f0e")
        plt.title("Articles per Month")
        plt.xlabel("Month")
        plt.ylabel("Article Count")
        plt.tight_layout()
        plt.savefig("plots/articles_per_month.png")
        if SHOW_PLOTS:
            plt.show()
    else:
        print("No valid dates in 'publish_date' to plot articles per month.")
else:
    print("Skipping articles-per-month line plot — 'publish_date' not found.")

# 3) Histogram: article lengths (word count)
# If your news items are in the same dataframe, we can still use 'word_count' as length proxy.
if "word_count" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df["word_count"].dropna(), bins=15, edgecolor="black", color="#9467bd")
    plt.title("Histogram of Article Lengths (Word Count)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/article_lengths_histogram.png")
    if SHOW_PLOTS:
        plt.show()
else:
    print("Skipping word count histogram — 'word_count' column not found.")

# ================================
#        TASK F — OPTIONAL ENHANCEMENTS
# ================================
print("\n=== TASK F: OPTIONAL ENHANCEMENTS ===")

# 1. Word count
texts = df["review_text"].fillna("").to_numpy()
df["word_count"] = np.array([len(t.split()) for t in texts])

print("=== Word Count Added ===")
print(df[["review_text", "word_count"]].head(), "\n")


# 2. Top keywords
texts_lower = df["review_text"].fillna("").str.lower().to_numpy()
stopwords = {"the", "and", "a", "to", "of", "in", "is", "it"}

all_words = []
for txt in texts_lower:
    all_words.extend([w for w in txt.split() if w not in stopwords])

keyword_counts = Counter(all_words)
top_keywords = keyword_counts.most_common(10)

print("=== Top 10 Keywords ===")
for word, count in top_keywords:
    print(f"{word}: {count}")
print()


# 3. Detect long & short reviews
counts = df["word_count"].to_numpy()
mean = np.mean(counts)
std = np.std(counts)

df["is_long"] = counts > (mean + 2 * std)
df["is_short"] = counts < (mean - 2 * std)

print("=== Long Reviews Detected ===")
print(df[df["is_long"]].head(), "\n")

print("=== Short Reviews Detected ===")
print(df[df["is_short"]].head(), "\n")


# 4. Sentiment proxy
ratings = df["rating"].to_numpy()
df["sentiment"] = np.where(
    ratings >= 7, "positive",
    np.where(ratings >= 4, "neutral", "negative")
)

print("=== Sentiment Proxy Added ===")
print(df[["rating", "sentiment"]].head())

# ================================
#        TASK G — Results Export
# ================================

# G.1 Export the cleaned and enhanced dataset to a new CSV file
OUTPUT_FILE = "data\\cleaned_data.csv"
print(df)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Enhanced dataset exported to: {OUTPUT_FILE}")

# G.2 Save results table → summary.csv
# --- Build summary table (key metrics) ---
summary_rows = []

# Basic rating stats
ratings = df["rating"].to_numpy()
summary_rows.extend([
    {"metric": "count_reviews", "value": int(np.size(ratings))},
    {"metric": "rating_mean", "value": float(np.mean(ratings)) if ratings.size else None},
    {"metric": "rating_median", "value": float(np.median(ratings)) if ratings.size else None},
    {"metric": "rating_min", "value": float(np.min(ratings)) if ratings.size else None},
    {"metric": "rating_max", "value": float(np.max(ratings)) if ratings.size else None},
])

# Optional: word count stats if available
if "word_count" in df.columns:
    wc = df["word_count"].to_numpy()
    summary_rows.extend([
        {"metric": "word_count_mean", "value": float(np.mean(wc)) if wc.size else None},
        {"metric": "word_count_std", "value": float(np.std(wc)) if wc.size else None},
        {"metric": "word_count_p95", "value": float(np.percentile(wc, 95)) if wc.size else None},
    ])

# Optional: sentiment distribution if available
if "sentiment" in df.columns:
    sentiment_counts = df["sentiment"].value_counts(dropna=False)
    for label, cnt in sentiment_counts.items():
        summary_rows.append({"metric": f"sentiment_{label}_count", "value": int(cnt)})

# Optional: number of unique movies
if "movie_title" in df.columns:
    summary_rows.append({"metric": "unique_movies", "value": int(df["movie_title"].nunique())})

# Create a tidy summary table and save as CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("data/summary.csv", index=False)
print("Saved: data/summary.csv")
