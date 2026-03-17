import pandas as pd
import matplotlib.pyplot as plt

# 1) Load and isnpect data

df = pd.read_csv(r"c:\Users\z233482\OneDrive - ZF Friedrichshafen AG\Documents\Miniprojekt\codingChallenge\movieReviews\movie_reviews.csv")

# Inspect first 10 lines of the DataFrame
print("\nFirst 10 lines:")
print(df.head(10))

# Shape of the DataFrame
print("\nShape of the DataFrame:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Data types
print("\nData types:")
print(df.info())

# Identify missing values
print("\nMissing values:")
print(df.isnull().sum())

# Identify duplicates
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 2) Data cleaning
print("\n##### 2 Data Cleaning #####\n")

# Remove duplicate rows
df = df.drop_duplicates()

#Converting dates using pd.to_datetime()
print("Converting date columns to datetime:")
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")


# Filling missing text fields with empty strings
text_columns = df.select_dtypes(include=["object", "string"]).columns
df[text_columns] = df[text_columns].fillna("")

# Trimming whitespace in text fields
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# Ensuring numeric fields (e.g., rating) have correct data types
numeric_cols = ["rating", "review_length", "word_count", "sentiment_score", "views"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3 Basic Statistical Analysis
print("\n##### 3 Basic Statistical Analysis #####\n")

# Movie review statistics
 # Mean, median, min and max rating
print("Rating statistics:")
rating_stats = {
    "mean": df["rating"].mean(),
    "median": df["rating"].median(),
    "min": df["rating"].min(),
    "max": df["rating"].max()
}
print(rating_stats)

# Number of reviews per movie
print("\nNumber of reviews per movie:\n")
reviews_per_movie = df["movie_title"].value_counts()
print(reviews_per_movie)

# Rating distribution
print("\nRating distribution:\n")
rating_distribution = df["rating"].value_counts().sort_index()
print(rating_distribution)

# Articels per category:
print("\nNumber of reviews per category:\n")
reviews_per_category = df["category"].value_counts()
print(reviews_per_category)

# Articals per author:
print("\nNumber of reviews per author:\n")
reviews_per_author = df["author"].value_counts()
print(reviews_per_author)

# Most common public day

print("\nMost common publish day:\n")

most_common_publish_day = df["publish_date"].dt.day_name()
print(most_common_publish_day.value_counts())

# Descriptive statistics for numeric columns
print("\nDescriptive statistics for numeric columns:\n")
print(df.describe())

# 4. Filtering & Grouping
print("\n##### 4 Filtering & Grouping #####\n")

# Filter reviews with rating >= 8
print("Reviews with rating >= 8:\n")
highly_rated_reviews = df[df["rating"] >= 8]
highly_rated_reviews = highly_rated_reviews.sort_values(by=["rating","movie_title"],ascending=[False,True])
print(highly_rated_reviews[["movie_title", "rating", "review_text"]].head(10))

# Top 5 most reviewed movies
print("\nTop 5 most reviewed movies:\n")
top5 = (
    df.groupby("movie_title")["review_id"]
      .nunique()                     # počet recenzí na film
      .reset_index(name="review_count")
      .sort_values("review_count", ascending=False)
      .reset_index(drop=True)
      .head(5)
)
top5.index = top5.index + 1 # Start index at 1
print(top5)

# Average rating per movie
print("\nAverage rating per movie:\n")
avg_rating = (
    df.groupby("movie_title")["rating"]
      .mean()
      .reset_index(name="average_rating")
      .sort_values("average_rating", ascending=False)
      .reset_index(drop=True)
)
avg_rating.index = avg_rating.index + 1 # Start index at 1
print(avg_rating)

# 5. Data Visualization (Matplotlib)
print("\n##### 5 Data Visualization #####\n")

#Histogram -> rating distribution
print("Rating distribution histogram:\n")

plt.figure(figsize=(6,4))
plt.hist(df["rating"], bins=10, color="#4f81bd", edgecolor="black")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.3)
plt.show()