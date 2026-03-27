"""
Sentiment Analysis Pipeline - Step 1 & Step 2
---------------------------------------------

This module implements the first two steps of a complete
sentiment‑analysis pipeline for movie review texts.

Step 1:
    Load the input CSV file `movie_reviews.csv`
    containing one movie-review sentence per row.

Step 2:
    Clean and preprocess the text to prepare it
    for tokenization and later ML model input.

Cleaning operations include:
    - Lowercasing
    - Removing HTML tags
    - Removing non-alphanumeric characters (except .,!?)
    - Normalizing whitespace
    - Stripping leading/trailing spaces

Input:
    movie_reviews.csv
        Column: review -> str, one sentence per row

Output:
    DataFrame `df_clean` containing cleaned text
"""

import re
import pandas as pd

# ---------------------------------------------------
# Step 1: Load input CSV file
# ---------------------------------------------------
INPUT_FILE = "movie_reviews.csv"

try:
    df = pd.read_csv(INPUT_FILE)
    print("CSV file loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found in the current directory.")
    raise

print("\nOriginal data preview:")
print(df.head())

# ---------------------------------------------------
# Step 2: Text Cleaning & Preprocessing
# ---------------------------------------------------

def clean_text(text: str) -> str:
    """Apply basic NLP cleaning to a text string."""
    text = text.lower()                                     # lowercase
    text = re.sub(r"<[^>]+>", " ", text)                    # remove HTML tags
    text = re.sub(r"[^a-z0-9.,!?\\s]", " ", text)           # remove unwanted characters
    text = re.sub(r"\\s+", " ", text)                       # collapse spaces
    return text.strip()                                     # strip whitespace

df["clean_review"] = df["review"].apply(clean_text)

print("\nCleaned data preview:")
print(df.head())

df_clean = df.copy()
