import pandas as pd
import os
from pathlib import Path
import re

import requests

#Step 1 — Load Input File
# 1.1 Validate that the file exists

if not os.path.isfile('movie_reviews.csv'):
    print("Error: 'movie_reviews.csv' not found in the current directory.")

# Load movie_reviews.csv into a pandas DataFrame.
df = pd.read_csv('movie_reviews.csv')
# 1.2 Show the first few lines for confirmation
#print(df.head())


#Step 2 — Clean and Preprocess Text
# 2.1 lowercases the reviews for uniformity.
df['review'] = df['review'].str.lower()
# 2.2 removes HTML tags using regex.
df['review'] = df['review'].str.replace(r'<.*?>', '', regex=True)
# 2.3 removes non‑alphanumeric characters except punctuation (. , ! ?)
df['review'] = df['review'].str.replace(r'[^a-zA-Z0-9.,!? ]', '', regex=True)
# 2.4 collapses multiple spaces
df['review'] = df['review'].str.replace(r'\s+', ' ', regex=True).str.strip()
# 2.5 trims the result
#print(df.head())
# Output: add a new column: clean_review
df['clean_review'] = df['review']
#print(df.head())

#Step 3 — Tokenization Using BERT Tokenizer
from transformers import BertTokenizer
# 3.1 By default, run fully offline without external downloads.
# Set TOKENIZER_MODE=bert_local and HF_MODEL_DIR=<local model path> to use local BERT files.
def simple_tokenize(text: str):
    # Split words and keep punctuation tokens similar to basic NLP preprocessing.
    return re.findall(r"[a-zA-Z0-9]+|[.,!?]", text)


tokenizer_mode = os.getenv('TOKENIZER_MODE', 'bert_local')
local_model_dir = os.getenv('HF_MODEL_DIR', os.getenv('HF_SENTIMENT_MODEL_DIR', 'local_models/textattack-bert-base-uncased-imdb'))

if tokenizer_mode == 'bert_local':
    if not local_model_dir:
        raise RuntimeError(
            "TOKENIZER_MODE=bert_local requires HF_MODEL_DIR pointing to a local BERT tokenizer directory."
        )

    try:
        tokenizer = BertTokenizer.from_pretrained(Path(local_model_dir), local_files_only=True)
        df['tokens'] = df['clean_review'].apply(tokenizer.tokenize)
    except (requests.exceptions.SSLError, OSError, ValueError) as exc:
        raise RuntimeError(
            "Failed to load local BERT tokenizer. Ensure HF_MODEL_DIR contains tokenizer files "
            "(vocab.txt, tokenizer_config.json, special_tokens_map.json)."
        ) from exc
else:
    df['tokens'] = df['clean_review'].apply(simple_tokenize)

#print(df.head())

#Step 4 — Convert Tokens to Numerical IDs
#Output: new column: token_ids
if tokenizer_mode == 'bert_local':
    df['token_ids'] = df['tokens'].apply(tokenizer.convert_tokens_to_ids)
else:
    # For simple tokenization, create a basic vocabulary mapping
    vocab = {token: idx for idx, token in enumerate(set(token for tokens in df['tokens'] for token in tokens))}
    df['token_ids'] = df['tokens'].apply(lambda tokens: [vocab[token] for token in tokens])

#print(df.head())

#Step 5 — Padding & Attention Masks
'''
Encode each review into:

padded input_ids
attention_mask
All sequences must have the same fixed length (e.g., 64 tokens).

Output: two new columns:

input_ids
attention_mask
'''
max_length = 64

def pad_and_create_attention_mask(token_ids):
    # 1) Truncate so we have space for [CLS] and [SEP]
    token_ids = token_ids[: max_length - 2]

    # 2) Add special tokens explicitly
    input_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

    # 3) Padding
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

    # 4) Attention mask (1 for real tokens incl. [CLS]/[SEP], 0 for padding)
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len

    return input_ids, attention_mask

df[['input_ids', 'attention_mask']] = df['token_ids'].apply(
    lambda ids: pd.Series(pad_and_create_attention_mask(ids))
)

#print(df.head())

#Step 6 — Run a Pretrained BERT Sentiment Classifier
'''
Use the pretrained model:

textattack/bert-base-uncased-imdb
For each review, run the model to obtain:

raw sentiment (positive/negative)
probability vector from softmax
Output: two columns:

sentiment_raw
probabilities
'''
from transformers import BertForSequenceClassification, BertTokenizer

local_sentiment_model_dir = os.getenv('HF_SENTIMENT_MODEL_DIR', 'local_models/textattack-bert-base-uncased-imdb')
model_path = Path(local_sentiment_model_dir)

if not model_path.exists():
    raise RuntimeError(
        f"Local model directory not found: {model_path}. "
        "Copy the sentiment model files to this folder or set HF_SENTIMENT_MODEL_DIR."
    )

try:
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
except (requests.exceptions.SSLError, OSError, ValueError) as exc:
    raise RuntimeError(
        "Failed to load local sentiment model. Ensure directory contains model files like "
        "config.json, pytorch_model.bin (or model.safetensors), vocab.txt, tokenizer_config.json."
    ) from exc
import torch

def predict_sentiment(input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        sentiment_raw = 'positive' if probabilities[1] > probabilities[0] else 'negative'
    return sentiment_raw, probabilities
df[['sentiment_raw', 'probabilities']] = df.apply(lambda row: pd.Series(predict_sentiment(row['input_ids'], row['attention_mask'])), axis=1)
#print(df.head())

#Step 7 — Produce Final Sentiment Predictions
'''
For each review:

choose the higher‑confidence label
compute confidence as a percentage
format final output like:
positive (97.44%)
Output: a new column: final_sentiment

Also generate:

✔ predictions.csv
Contains:

original review
final sentiment prediction
✔ result.csv
Contains all intermediate steps
(Useful for debugging or teaching.)
'''
def format_final_sentiment(sentiment_raw, probabilities):
    confidence = max(probabilities) * 100
    return f"{sentiment_raw} ({confidence:.2f}%)"
df['final_sentiment'] = df.apply(lambda row: format_final_sentiment(row['sentiment_raw'], row['probabilities']), axis=1)
# Save predictions.csv with original review and final sentiment prediction
df[['review', 'final_sentiment']].to_csv('predictions.csv', index=False)
# Save result.csv with all intermediate steps
df.to_csv('result.csv', index=False)

print(df.head())