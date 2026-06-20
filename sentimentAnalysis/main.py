"""
Entry point for running the sentiment analysis pipeline.

Usage:
    python main.py
"""

from sentiment_pipeline import SentimentPipeline


if __name__ == "__main__":
    pipeline = SentimentPipeline(
        input_file="movie_reviews.csv",
        predictions_file="predictions.csv",
        output_file="result.csv",
    )
    pipeline.run()
