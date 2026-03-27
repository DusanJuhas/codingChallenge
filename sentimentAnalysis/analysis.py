"""
Sentiment Analysis Pipeline - Steps 1 to 7 (Practical Version)
-------------------------------------------------------------

This script implements a practical, real-world BERT-based
sentiment-analysis pipeline for movie reviews.

Step 1:
    Load the input CSV file.

Step 2:
    Clean and preprocess the text.

Step 3:
    Tokenize cleaned text using BERT WordPiece tokenizer.

Step 4:
    Convert tokens to numerical token IDs.

Step 5:
    Apply padding and create attention masks.

Step 6:
    Run a pretrained BERT sentiment classifier (IMDB model).

Step 7:
    Produce final sentiment predictions with confidence
    and save the results to CSV files.

The output includes:
- result.csv: full internal data (debug/log format)
- predictions.csv: clean human-readable sentiment labels
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
    Tokenize text into WordPiece tokens using BERT tokenizer.
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
    Encode text using the BERT tokenizer, generating:
    - input_ids (padded)
    - attention_mask (1=token, 0=padding)
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
# Step 6: Real BERT Sentiment Classification (IMDB model)
# ---------------------------------------------------
print("\nLoading BERT IMDB sentiment model...")
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb"
)
model.eval()


def classify_sentiment(input_ids, attention_mask):
    """
    Perform inference using the pretrained IMDB BERT classifier.
    Returns the raw sentiment label and probability vector.
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


df["sentiment_raw"], df["probabilities"] = zip(
    *df.apply(
        lambda row: classify_sentiment(row["input_ids"], row["attention_mask"]),
        axis=1,
    )
)

print("\nRaw sentiment prediction preview:")
print(df[["clean_review", "sentiment_raw", "probabilities"]].head())


# ---------------------------------------------------
# Step 7: Produce final clean sentiment output
# ---------------------------------------------------
def final_prediction(label, probabilities):
    """
    Format the final sentiment label with a confidence percentage.
    """
    positive_prob = probabilities[1]
    negative_prob = probabilities[0]
    confidence = round(max(positive_prob, negative_prob) * 100, 2)
    return f"{label} ({confidence}%)"


df["final_sentiment"] = df.apply(
    lambda row: final_prediction(row["sentiment_raw"], row["probabilities"]),
    axis=1
)

print("\nFinal sentiment prediction preview:")
print(df[["clean_review", "final_sentiment"]].head())


# Save simplified predictions file
PRED_FILE = "predictions.csv"
df[["review", "final_sentiment"]].to_csv(PRED_FILE, index=False)
print(f"\nHuman-readable predictions saved to {PRED_FILE}")


# Save the full dataset for debugging
OUTPUT_FILE = "result.csv"
df.to_csv(OUTPUT_FILE, index=False)
print(f"Full result saved to {OUTPUT_FILE}")
