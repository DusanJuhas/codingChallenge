# import c:\temp\VS\Movie_Reviews_Dataset_UTF8.csv 

import re
from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt
import os

# A. Load & Inspect Data
class MovieReviewAnalyser:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def average_rating_by_genre(self):
        return self.data.groupby('genre')['rating'].mean() #Groups the dataset by genre and calculates the average rating for each genre.

    def display_first_n_rows(self, n):
        return self.data.head(n)
    
    def remove_duplicate_rows(self):
        self.data.drop_duplicates(inplace=True)
        print("B.1 Data Cleaning - Remove duplicates Duplicate rows ... removed successfully.")

    def handle_missing_values(self):
        print("B.2 Handling missing values...")
        self.data.fillna('', inplace=True)

    def convert_date_columns(self, date_columns):
        print("B.3 Data Cleaning - Convert date columns using pd.to_datetime()")
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                print(f"Converted {col}, dtype: {self.data[col].dtype}")
        else:
            print(f"Column {col} not found in data!")
    
    def trim_whitespace(self, text_columns):
        print("B.4 Data Cleaning - Trim whitespace in text fields")
        for col in text_columns:
            self.data[col] = self.data[col].str.strip()
    
    def convert_numeric_columns(self, numeric_columns):
        print("B.5 Data Cleaning - Convert numeric columns to correct dtype")
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

    def clean_data(self):
        self.remove_duplicate_rows()
        self.handle_missing_values()
        self.convert_date_columns(['publish_date', 'review_date'])
        self.trim_whitespace(['movie_title', 'review_text'])
        self.convert_numeric_columns(['rating', 'review_length', 'word_count', 'sentiment_score', 'views'])

    def data_analysis(self):
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
        return self.data[self.data['rating'] >= threshold]
    
    def top_n_most_reviewed_movies(self, n):
        return self.data['movie_title'].value_counts().head(n)
    
    def average_rating_per_movie(self):
        return self.data.groupby('movie_title')['rating'].mean()
    
    def articles_in_category(self, category):
        return self.data[self.data['category'] == category]
    
    def most_active_author(self):
        return self.data['author'].value_counts().idxmax()
    
    def number_of_articles_per_month(self):
        return self.data['publish_date'].dt.to_period('M').value_counts().sort_index()
    
    def extract_keywords(self, column, top_n=10):
        all_words = self.data[column].dropna().astype(str).str.lower().str.cat(sep=' ')
        words = re.findall(r'\b\w+\b', all_words)
        common_words = Counter(words).most_common(top_n)
        return [word for word, _ in common_words]
    
    def detect_extremely_long_or_short_reviews(self, min_length=10, max_length=500):
        long_reviews = self.data[self.data['review_length'] > max_length]
        short_reviews = self.data[self.data['review_length'] < min_length]
        return long_reviews, short_reviews
    
    def create_sentiment_proxy(self, positive_threshold=7, negative_threshold=4):
        self.data['sentiment'] = 'Neutral'
        self.data.loc[self.data['rating'] >= positive_threshold, 'sentiment'] = 'Positive'
        self.data.loc[self.data['rating'] <= negative_threshold, 'sentiment'] = 'Negative'



# prepare folder structure
'''
project/
│── data/
│    └── original.csv
│    └── cleaned_data.csv
│    └── summary.csv
│
│── plots/
│    └── rating_histogram.png
│    └── reviews_over_time.png
│
│── movie_review.py
│── README.md
'''
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('data'):
    os.makedirs('data')

sFileToAnalyse = r'data/original.csv'


# A.1 Load the CSV using pandas.read_csv()
#analyser = MovieReviewAnalyser(r'c:\temp\VS\Movie_Reviews_Dataset_UTF8.csv')
analyser = MovieReviewAnalyser(sFileToAnalyse)

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
    def __init__(self, file_path):
        super().__init__(file_path)
    
    def plot_rating_distribution(self):
        plt.figure(figsize=(10, 6))
        self.data['rating'].hist(bins=10, edgecolor='black')
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        #plt.show()
        plt.savefig('plots/rating_distribution.png')

    def plot_average_rating_per_movie(self):
        average_rating_per_movie = self.data.groupby('movie_title')['rating'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        average_rating_per_movie.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Average Rating per Movie')
        plt.xlabel('Movie Title')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        #plt.show()
        plt.savefig('plots/average_rating_per_movie.png')
    
    def plot_reviews_over_time(self):
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
        self.data['publish_date'] = pd.to_datetime(self.data['publish_date'], errors='coerce')
        mask = self.data['publish_date'].notna()
        articles_per_month = self.data.loc[mask].groupby(self.data.loc[mask, 'publish_date'].dt.to_period('M')).size()
        plt.figure(figsize=(12, 6))
        articles_per_month.plot(kind='line', marker='o', color='skyblue')
        plt.title('Articles per Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        #plt.show()
        plt.savefig('plots/articles_per_month.png')

    def plot_article_length_distribution(self):
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



