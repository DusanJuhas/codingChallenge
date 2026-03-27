"""
Sentiment Analysis Pipeline - Step 1, Step 2, Step 3
----------------------------------------------------

This module implements the first three steps of a complete
sentiment-analysis pipeline for movie review texts.

Step 1:
    Load the input CSV file `movie_reviews.csv`.

Step 2:
    Clean and preprocess the text.

Step 3:
    Tokenize the cleaned text using the BERT tokenizer
    (WordPiece subword tokenization).

Input:
    movie_reviews.csv
        Column: review -> str

Output:
    DataFrame `df_tokens` with tokenized text.
"""

import re
import pandas as pd
from transformers import BertTokenizer

# ---------------------------------------------------
# Step 1: Load input CSV file
# ---------------------------------------------------
INPUT_FILE = "movie_reviews.csv"

try:
    df = pd.read_csv(INPUT_FILE)
    print("CSV file loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found.")
    raise

print("\nOriginal data preview:")
print(df.head())

# ---------------------------------------------------
# Step 2: Text Cleaning & Preprocessing
# ---------------------------------------------------

def clean_text(text: str) -> str:
    """Apply basic NLP cleaning to a text string."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)              # remove HTML tags
    text = re.sub(r"[^a-z0-9.,!?\\s]", " ", text)     # remove unwanted chars
    text = re.sub(r"\\s+", " ", text)                 # collapse spaces
    return text.strip()

df["clean_review"] = df["review"].apply(clean_text)

print("\nCleaned data preview:")
print(df.head())

# ---------------------------------------------------
# Step 3: Tokenization using BERT tokenizer
# ---------------------------------------------------

# Load tokenizer (bert-base-uncased is standard for English sentiment tasks)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text: str):
    """Tokenize text into WordPiece tokens using BERT tokenizer."""
    return tokenizer.tokenize(text)

df["tokens"] = df["clean_review"].apply(tokenize_text)

print("\nTokenized data preview:")
print(df[["clean_review", "tokens"]].head())

# Store result
df_tokens = df.copy()
