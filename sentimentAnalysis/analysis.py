"""
Sentiment Analysis Pipeline - Steps 1 to 6
------------------------------------------

This module implements a sentiment-analysis pipeline using BERT.
Steps included:

1. Load input CSV file.
2. Clean and preprocess the text.
3. Tokenize using a BERT WordPiece tokenizer.
4. Convert tokens to numerical IDs.
5. Apply padding and create attention masks.
6. Run an IMDB-trained BERT classifier to predict sentiment.

The output is stored in result.csv.
"""

import re
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------------------
# Step 1: Load input CSV file
# ---------------------------------------------------
INPUT_FILE = "movie_reviews.csv"

try:
    df = pd.read_csv(INPUT_FILE)
    print("CSV file loaded successfully!")
except FileNotFoundError as exc:
    print(f"Error: The file '{INPUT_FILE}' was not found.")
    raise exc

print("\nOriginal data preview:")
print(df.head())


# ---------------------------------------------------
# Step 2: Text Cleaning & Preprocessing
# ---------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean review text by:
    - lowercasing
    - removing HTML tags
    - removing unwanted characters
    - collapsing whitespace
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9.,!?\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


df["clean_review"] = df["review"].apply(clean_text)

print("\nCleaned data preview:")
print(df.head())


# ---------------------------------------------------
# Step 3: Tokenization using BERT tokenizer
# ---------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_text(text: str):
    """
    Tokenize text into BERT WordPiece tokens.
    """
    return tokenizer.tokenize(text)


df["tokens"] = df["clean_review"].apply(tokenize_text)

print("\nTokenized data preview:")
print(df[["clean_review", "tokens"]].head())


# ---------------------------------------------------
# Step 4: Convert tokens to numerical IDs
# ---------------------------------------------------
def tokens_to_ids(tokens):
    """
    Convert WordPiece tokens into BERT vocabulary IDs.
    """
    return tokenizer.convert_tokens_to_ids(tokens)


df["token_ids"] = df["tokens"].apply(tokens_to_ids)

print("\nToken ID preview:")
print(df[["tokens", "token_ids"]].head())


# ---------------------------------------------------
# Step 5: Padding & Attention Masks
# ---------------------------------------------------
MAX_LEN = 64


def encode_with_padding(text: str):
    """
    Encode text using BERT tokenizer with:
    - special tokens
    - fixed-length padding
    - attention masks
    Returns: (input_ids, attention_mask)
    """
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()
    return input_ids, attention_mask


df["input_ids"], df["attention_mask"] = zip(
    *df["clean_review"].apply(encode_with_padding)
)

print("\nInput IDs and Attention Mask preview:")
print(df[["clean_review", "input_ids", "attention_mask"]].head())


# ---------------------------------------------------
# Step 6: Real BERT Model Sentiment Classification
# ---------------------------------------------------
print("\nLoading BERT sentiment model...")
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb"
)
model.eval()


def classify_sentiment(input_ids, attention_mask):
    """
    Perform inference using a pretrained IMDB BERT classifier.
    Returns:
    - sentiment label ("positive" or "negative")
    - probability vector
    """
    ids_tensor = torch.tensor(input_ids).unsqueeze(0)
    mask_tensor = torch.tensor(attention_mask).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=ids_tensor, attention_mask=mask_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]

    sentiment_idx = int(torch.argmax(logits, dim=1).item())
    sentiment_label = "positive" if sentiment_idx == 1 else "negative"

    return sentiment_label, probabilities


df["sentiment"], df["probabilities"] = zip(
    *df.apply(
        lambda row: classify_sentiment(row["input_ids"], row["attention_mask"]),
        axis=1,
    )
)

print("\nSentiment prediction preview:")
print(df[["clean_review", "sentiment", "probabilities"]].head())


# ---------------------------------------------------
# Save results to CSV
# ---------------------------------------------------
df_final = df.copy()

OUTPUT_FILE = "result.csv"
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nResult saved to {OUTPUT_FILE}")
