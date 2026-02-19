import pandas as pd
import matplotlib.pyplot as plt

# csv_file is the path to the CSV file containing the film reviews dataset
csv_file = "data/original.csv"

# row_count is the number of rows to display when showing the first few rows of the dataset
row_count = 10

try:
# task A.1 → Load the dataset
    df = pd.read_csv(csv_file)
    print(f"First {row_count} rows of the dataset:\n")
# task A.2.a → Display first 10 rows
    print(df.head(row_count))
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
    print(f"Error: File not found at path '{csv_file}'.")
    quit()
except Exception as e:
    print(f"An error occurred: {e}")
    quit()

# task B.1 → remove duplicates
df = df.drop_duplicates().reset_index(drop=True) 
print(f"\n=== Shape after duplicate removal: {df.shape} ===")

# task B.2 → handle missing values by filling with empty string
# empty_string is a variable to be used to fill missing values in the dataset 
empty_string = ""
df = df.fillna(empty_string)
print(f"\nMissing values FILLED with: '{empty_string}'")

# task B.3 → convert date columns to datetime
# datafield name for date in the dataset
date_column = "review_date"
df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
print(f"Converted '{date_column}' to datetime.")

# task B.4 → Trim whitespace in text fields
# text_columns is a list of column names that contain text data and may require trimming and normalization
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
