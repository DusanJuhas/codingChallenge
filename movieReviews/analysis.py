"""
Movie Review Analyzer
---------------------
This module provides tools for loading, cleaning, analyzing, 
and visualizing movie review and news datasets.
It uses pandas for data manipulation and matplotlib for visualization.
"""
import os
import re
from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt

#set CWD
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# A. Load & Inspect Data
class MovieReviewAnalyser:
    """
    Class for analyzing and cleaning movie review and news datasets.
    Provides methods for data loading, cleaning, analysis, and feature extraction.
    """
    def __init__(self, file_path):
        """
        Initialize the MovieReviewAnalyser with a CSV file path.
        Loads the data into a pandas DataFrame.
        Args:
            file_path (str): Path to the CSV file.
        """
        self.data = pd.read_csv(file_path)

    def average_rating_by_genre(self):
        """
        Groups the dataset by genre and calculates the average rating for each genre.
        Returns:
            pandas.Series: Average rating for each genre.
        """
        return self.data.groupby('genre')['rating'].mean()

    def display_first_n_rows(self, n):
        """
        Display the first n rows of the dataset.
        Args:
            n (int): Number of rows to display.
        Returns:
            pandas.DataFrame: First n rows of the dataset.
        """
        return self.data.head(n)

    def remove_duplicate_rows(self):
        """
        Remove duplicate rows from the dataset in place.
        """
        self.data.drop_duplicates(inplace=True)
        print("B.1 Data Cleaning - Remove duplicates Duplicate rows ... removed successfully.")

    def handle_missing_values(self):
        """
        Fill missing values in the dataset with empty strings.
        """
        print("B.2 Handling missing values...")
        self.data.fillna('', inplace=True)

    def convert_date_columns(self, date_columns):
        """
        Convert specified columns to datetime using pandas to_datetime.
        Args:
            date_columns (list): List of column names to convert.
        """
        print("B.3 Data Cleaning - Convert date columns using pd.to_datetime()")
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                print(f"Converted {col}, dtype: {self.data[col].dtype}")
            else:
                print(f"Column {col} not found in data!")

    def trim_whitespace(self, text_columns):
        """
        Trim whitespace from specified text columns.
        Args:
            text_columns (list): List of column names to trim.
        """
        print("B.4 Data Cleaning - Trim whitespace in text fields")
        for col in text_columns:
            self.data[col] = self.data[col].str.strip()

    def convert_numeric_columns(self, numeric_columns):
        """
        Convert specified columns to numeric dtype.
        Args:
            numeric_columns (list): List of column names to convert.
        """
        print("B.5 Data Cleaning - Convert numeric columns to correct dtype")
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

    def clean_data(self):
        """
        Perform all data cleaning steps: remove duplicates, handle missing values,
        convert date columns, trim whitespace, and convert numeric columns.
        """
        self.remove_duplicate_rows()
        self.handle_missing_values()
        self.convert_date_columns(['publish_date', 'review_date'])
        self.trim_whitespace(['movie_title', 'review_text'])
        self.convert_numeric_columns(['rating', 'review_length', 'word_count', 'sentiment_score', 'views'])

    def data_analysis(self):
        """
        Perform various data analysis tasks and print results, including statistics,
        counts, and distributions for movie reviews and news articles.
        """
        # C.1.1 Data Analysis - Movie reviews: Mean, median, min, max rating
        print("C.1.1 Movie reviews - Rating statistics:")
        print("Mean rating:", self.data['rating'].mean())
        print("Median rating:", self.data['rating'].median())
        print("Min rating:", self.data['rating'].min())
        print("Max rating:", self.data['rating'].max())
        # C.1.2 Data Analysis - Movie reviews: Count of reviews per movie
        print("C.1.2 Movie reviews - Count of reviews per movie:")
        print(self.data['movie_title'].value_counts())

        # C.1.3 Data Analysis - Movie reviews: Distribution of ratings (e.g. matplotlib graph)
        plt.figure(figsize=(10, 6))
        self.data['rating'].hist(bins=10, edgecolor='black')
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        #plt.show()
        # save to folder plots/
        plt.savefig('plots/distribution_of_movie_ratings.png')

        # C.2.1 Data Analysis - News: Count of articles per category
        print("C.2.1 News - Count of articles per category:")
        print(self.data['category'].value_counts())
        # C.2.2 Data Analysis - News: Articles per author
        print("C.2.2 News - Articles per author:")
        print(self.data['author'].value_counts())
        # C.2.3 Data Analysis - News: Most frequent publish day
        print("C.2.3 News - Most frequent publish day:")
        print(self.data['publish_date'].dt.day_name().value_counts().idxmax())
        # C.2.4 Data Analysis - News: Summary statistics for numeric columns
        print("C.2.4 News - Summary statistics for all columns of type numeric:")
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        print(self.data[numeric_columns].describe())

    def filter_reviews_by_rating(self, threshold):
        """
        Filter reviews with rating greater than or equal to the threshold.
        Args:
            threshold (float): Minimum rating value.
        Returns:
            pandas.DataFrame: Filtered reviews.
        """
        return self.data[self.data['rating'] >= threshold]

    def top_n_most_reviewed_movies(self, n):
        """
        Get the top n most reviewed movies.
        Args:
            n (int): Number of top movies to return.
        Returns:
            pandas.Series: Movie titles and their review counts.
        """
        return self.data['movie_title'].value_counts().head(n)

    def average_rating_per_movie(self):
        """
        Calculate the average rating for each movie.
        Returns:
            pandas.Series: Average rating per movie.
        """
        return self.data.groupby('movie_title')['rating'].mean()

    def articles_in_category(self, category):
        """
        Get all articles in a specified category.
        Args:
            category (str): Category name.
        Returns:
            pandas.DataFrame: Articles in the category.
        """
        return self.data[self.data['category'] == category]

    def most_active_author(self):
        """
        Find the author with the most articles.
        Returns:
            str: Author name.
        """
        return self.data['author'].value_counts().idxmax()

    def number_of_articles_per_month(self):
        """
        Count the number of articles published each month.
        Returns:
            pandas.Series: Number of articles per month.
        """
        return self.data['publish_date'].dt.to_period('M').value_counts().sort_index()

    def extract_keywords(self, column, top_n=10):
        """
        Extract the top N most common keywords from a text column.
        Args:
            column (str): Column name to extract keywords from.
            top_n (int): Number of top keywords to return.
        Returns:
            list: List of top keywords.
        """
        all_words = self.data[column].dropna().astype(str).str.lower().str.cat(sep=' ')
        words = re.findall(r'\b\w+\b', all_words)
        common_words = Counter(words).most_common(top_n)
        return [word for word, _ in common_words]

    def detect_extremely_long_or_short_reviews(self, min_length=10, max_length=500):
        """
        Detect reviews that are extremely long or short based on word count.
        Args:
            min_length (int): Minimum word count for a review to be considered short.
            max_length (int): Maximum word count for a review to be considered long.
        Returns:
            tuple: (long_reviews, short_reviews) DataFrames.
        """
        long_reviews_local = self.data[self.data['review_length'] > max_length]
        short_reviews_local = self.data[self.data['review_length'] < min_length]
        return long_reviews_local, short_reviews_local

    def create_sentiment_proxy(self, positive_threshold=7, negative_threshold=4):
        """
        Create a sentiment proxy column based on rating thresholds.
        Args:
            positive_threshold (float): Minimum rating for positive sentiment.
            negative_threshold (float): Maximum rating for negative sentiment.
        """
        self.data['sentiment'] = 'Neutral'
        self.data.loc[self.data['rating'] >= positive_threshold, 'sentiment'] = 'Positive'
        self.data.loc[self.data['rating'] <= negative_threshold, 'sentiment'] = 'Negative'



# prepare folder structure
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('data'):
    os.makedirs('data')

SFILE_TO_ANALYSE = r'data/original.csv'


# A.1 Load the CSV using pandas.read_csv()
analyser = MovieReviewAnalyser(SFILE_TO_ANALYSE)

# A.2a Display: First 10 rows
print("A.2a First 10 rows of the dataset:")
print(analyser.display_first_n_rows(10))

# A.2b Display: Dataset shape
print("A.2b Dataset shape:")
print(analyser.data.shape)

# A.2c Display: Column names
print("A.2c Column names:")
print(analyser.data.columns)

# A.2d Display: Data types using df.info()
print("A.2d Data types:")
print(analyser.data.info())

# A.3a Identify: Missing values
print("A.3a Missing values:")
print(analyser.data.isnull().sum())

# A.3b Identify: Duplicate rows
print("A.3b Duplicate rows:")
print(analyser.data.duplicated().sum())

# B Data Cleaning
print("B Data Cleaning")
analyser.clean_data()

# G.1 Export Results - Save cleaned dataset → cleaned_data.csv
print("G.1 Export Results - Save cleaned dataset to 'cleaned_data.csv'...")
analyser.data.to_csv('data/cleaned_data.csv', index=False)
print("Cleaned dataset saved successfully.")


# C Data Analysis
print("C Data Analysis")
analyser.data_analysis()

# G.2 Export Results - Save results table → summary.csv
print("G.2 Export Results - Save results to 'summary.csv'...")
#??


print("D.1.1 Filtering & Grouping - Movie reviews:Show reviews with rating ≥ 8:")
highly_rated_reviews = analyser.filter_reviews_by_rating(8)
print(highly_rated_reviews[['movie_title', 'rating', 'review_text']].head())

print("D.1.2 Filtering & Grouping - Movie reviews: Top 5 most-reviewed movies:")
top_5_movies = analyser.top_n_most_reviewed_movies(5)
print(top_5_movies)

print("D.2.1 Filtering & Grouping - Movie reviews: Average rating per movie:")
average_rating_per_movie = analyser.average_rating_per_movie()
print(average_rating_per_movie)

print("D.2.1 Filtering & Grouping - News: Articles in the “Technology” category:")
technology_articles = analyser.articles_in_category('Technology')
print(technology_articles)

print("D.2.2 Filtering & Grouping - News: Most active author:")
most_active_author = analyser.most_active_author()
print(most_active_author)

print("D.2.3 Filtering & Grouping - News: Number of articles published each month:")
articles_per_month = analyser.number_of_articles_per_month()
print(articles_per_month)

# E Data Visualization (Matplotlib)
class MovieVisualization(MovieReviewAnalyser):
    """
    Class for visualizing movie review and news datasets using matplotlib.
    Provides methods for various plots and charts.
    """
    def __init__(self, file_path):
        """
        Initialize the MovieVisualization with a CSV file path.
        Loads the data into a pandas DataFrame for visualization.
        Args:
            file_path (str): Path to the CSV file.
        """
        MovieReviewAnalyser.__init__(self, file_path)

    def plot_rating_distribution(self):
        """
        Plot and save a histogram showing the distribution of movie ratings.
        The plot is saved to 'plots/rating_distribution.png'.
        """
        plt.figure(figsize=(10, 6))
        self.data['rating'].hist(bins=10, edgecolor='black')
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        #plt.show()
        plt.savefig('plots/rating_distribution.png')

    def plot_average_rating_per_movie(self):
        """
        Plot and save a bar chart of the top 10 movies by average rating.
        The plot is saved to 'plots/average_rating_per_movie.png'.
        """
        avg_rating_per_movie = self.data.groupby('movie_title')['rating'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        avg_rating_per_movie.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Average Rating per Movie')
        plt.xlabel('Movie Title')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        #plt.show()
        plt.savefig('plots/average_rating_per_movie.png')

    def plot_reviews_over_time(self):
        """
        Plot and save a line chart showing the number of reviews over time (by month).
        The plot is saved to 'plots/reviews_over_time.png'.
        """
        self.data['review_date'] = pd.to_datetime(self.data['review_date'], errors='coerce')
        mask = self.data['review_date'].notna()
        reviews_over_time = self.data.loc[mask].groupby(self.data.loc[mask, 'review_date'].dt.to_period('M')).size()
        plt.figure(figsize=(12, 6))
        reviews_over_time.plot(kind='line', marker='o', color='skyblue')
        plt.title('Number of Reviews Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        #plt.show()
        plt.savefig('plots/reviews_over_time.png')

    def plot_articles_per_category(self):
        """
        Plot and save a bar chart showing the number of articles per category.
        The plot is saved to 'plots/articles_per_category.png'.
        """
        articles_per_category = self.data['category'].value_counts()
        plt.figure(figsize=(12, 6))
        articles_per_category.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Articles per Category')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        #plt.show()
        plt.savefig('plots/articles_per_category.png')

    def plot_articles_per_month(self):
        """
        Plot and save a line chart showing the number of articles published each month.
        The plot is saved to 'plots/articles_per_month.png'.
        """
        self.data['publish_date'] = pd.to_datetime(self.data['publish_date'], errors='coerce')
        mask = self.data['publish_date'].notna()
        articles_per_month_local = self.data.loc[mask].groupby(self.data.loc[mask, 'publish_date'].dt.to_period('M')).size()
        plt.figure(figsize=(12, 6))
        articles_per_month_local.plot(kind='line', marker='o', color='skyblue')
        plt.title('Articles per Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        #plt.show()
        plt.savefig('plots/articles_per_month.png')

    def plot_article_length_distribution(self):
        """
        Plot and save a histogram showing the distribution of article/review lengths (word count).
        The plot is saved to 'plots/article_length_distribution.png'.
        """
        plt.figure(figsize=(12, 6))
        self.data['review_length'].hist(bins=20, edgecolor='black')
        plt.title('Distribution of Article Lengths (Word Count)')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        #plt.show()
        plt.savefig('plots/article_length_distribution.png')

MovieVisual = MovieVisualization('data/cleaned_data.csv')

# E.1.1 Data Visualization (Matplotlib) - Movie reviews: Histogram: rating distribution
MovieVisual.plot_rating_distribution()
# E.1.2 Data Visualization (Matplotlib) - Movie reviews: Bar plot: average rating per movie
MovieVisual.plot_average_rating_per_movie()
# E.1.3 Data Visualization (Matplotlib) - Movie reviews: Line plot: number of reviews over time
MovieVisual.plot_reviews_over_time()
# E.2.1 Data Visualization (Matplotlib) - News: Bar chart: articles per category
MovieVisual.plot_articles_per_category()
# E.2.2 Data Visualization (Matplotlib) - News: Line plot: articles per month
MovieVisual.plot_articles_per_month()
# E.2.3 Data Visualization (Matplotlib) - News: Histogram: article lengths (word count)
MovieVisual.plot_article_length_distribution()

# F.1 Optional Advanced Tasks - Create a word-count column for each review/article and save it to cleaned_data.csv
print("F.1 Optional Advanced Tasks - Create a word-count column for each review/article:")
analyser.data['word_count'] = analyser.data['review_text'].apply(lambda x: len(str(x).split()))
analyser.data.to_csv('data/cleaned_data.csv', index=False)

# F.2 Optional Advanced Tasks - Extract top keywords using simple string operations
print("F.2 Optional Advanced Tasks - Extract top keywords using simple string operations:")
analyser.data['keywords'] = [analyser.extract_keywords('review_text', top_n=5)] * len(analyser.data)
print(analyser.data[['review_text', 'keywords']].head())


# F.3 Optional Advanced Tasks - Detect extremely long or short reviews/articles
print("F.3 Optional Advanced Tasks - Detect extremely long or short reviews/articles:")
long_reviews, short_reviews = analyser.detect_extremely_long_or_short_reviews(min_length=50, max_length=1000)
print("Long reviews/articles:")
print(long_reviews[['review_text', 'review_length']].head())
print("Short reviews/articles:")
print(short_reviews[['review_text', 'review_length']].head())

# F.4 Optional Advanced Tasks - Create a “sentiment proxy” (e.g., using rating thresholds)
print("F.4 Optional Advanced Tasks - Create a 'sentiment proxy' using rating thresholds:")
analyser.create_sentiment_proxy(positive_threshold=7, negative_threshold=4)
print(analyser.data[['rating', 'sentiment']].head())
