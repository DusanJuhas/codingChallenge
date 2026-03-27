"""
This script represents the first step of building a complete
sentiment‑analysis pipeline for movie review texts.

Purpose:
    - Load the input CSV file `movie_reviews.csv`
      containing one movie‑review sentence per row.
    - Prepare the dataset for further NLP preprocessing
      (cleaning, tokenization, embedding, etc.).

Input:
    movie_reviews.csv
        A CSV file with a single column:
            review  -> str, one sentence per row

Output:
    Prints confirmation that the data has been loaded
    and displays the first few rows of the dataset.
"""

import pandas as pd

# Step 1: Load input CSV file
INPUT_FILE = "movie_reviews.csv"

try:
    df = pd.read_csv(INPUT_FILE)
    print("CSV file loaded successfully!")
    print(df.head())  # Preview the first rows
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found in the current directory.")
