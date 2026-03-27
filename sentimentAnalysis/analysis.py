"""
Sentiment Analysis Pipeline - Steps 1 to 5
------------------------------------------

This module implements the first five steps of a complete
sentiment-analysis pipeline for movie review texts.

Step 1:
    Load the input CSV file `movie_reviews.csv`.

Step 2:
    Clean and preprocess the text.

Step 3:
    Tokenize cleaned text using the BERT tokenizer (WordPiece).

Step 4:
    Convert tokens to numerical token IDs.

Step 5:
    Apply padding and create attention masks.

Final:
    Store results into result.csv.
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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text: str):
    """Tokenize text into WordPiece tokens using BERT tokenizer."""
    return tokenizer.tokenize(text)

df["tokens"] = df["clean_review"].apply(tokenize_text)

print("\nTokenized data preview:")
print(df[["clean_review", "tokens"]].head())

# ---------------------------------------------------
# Step 4: Convert tokens to numerical IDs
# ---------------------------------------------------

def tokens_to_ids(tokens):
    """Convert a list of tokens into BERT vocabulary IDs."""
    return tokenizer.convert_tokens_to_ids(tokens)

df["token_ids"] = df["tokens"].apply(tokens_to_ids)

print("\nToken ID preview:")
print(df[["tokens", "token_ids"]].head())

# ---------------------------------------------------
# Step 5: Padding & Attention Masks (FIXED)
# ---------------------------------------------------

MAX_LEN = 64  # Adjust as needed

def encode_with_padding(text: str):
    """Encode text using BERT tokenizer with padding & attention masks."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"  # PyTorch tensors
    )

    return (
        encoding["input_ids"][0].tolist(),
        encoding["attention_mask"][0].tolist()
    )

df["input_ids"], df["attention_mask"] = zip(
    *df["clean_review"].apply(encode_with_padding)
)

print("\nInput IDs and Attention Mask preview:")
print(df[["clean_review", "input_ids", "attention_mask"]].head())

# Final dataset
df_ids = df.copy()

# ---------------------------------------------------
# Save results to CSV
# ---------------------------------------------------
OUTPUT_FILE = "result.csv"
df_ids.to_csv(OUTPUT_FILE, index=False)
print(f"\nResult saved to {OUTPUT_FILE}")