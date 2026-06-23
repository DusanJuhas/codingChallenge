import pandas as pd
from pathlib import Path
import requests
import torch
import sys
from bs4 import BeautifulSoup

#Step 1 — Load Input File
# 1.1 Validate that the files exists

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / 'movie_reviews.csv'
PREDICTIONS_CSV = SCRIPT_DIR / 'predictions.csv'
RESULT_CSV = SCRIPT_DIR / 'result.csv'
LOCAL_MODEL_DIR = Path(
    'c:/Users/Z128278/OneDrive - ZF Friedrichshafen AG/ZFOnedriveDocu/VS/miniProject_sentimentAnalyzer/local_models/textattack-bert-base-uncased-imdb'
)


if not INPUT_CSV.is_file():
    print(f"Error: Input file not found: {INPUT_CSV}")
    sys.exit(1)

# Load movie_reviews.csv into a pandas DataFrame.
df = pd.read_csv(INPUT_CSV)
# 1.2 Show the first few lines for confirmation
#print(df.head())


#Step 2 — Clean and Preprocess Text
# 2.1 lowercases the reviews for uniformity.
df['review'] = df['review'].str.lower()
# 2.2 removes HTML tags using BeautifulSoup.
df['review'] = df['review'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
# 2.3 removes non‑alphanumeric characters except punctuation (. , ! ? ') and apostrophes for contractions.
df['review'] = df['review'].str.replace(r"[^a-zA-Z0-9.,!?' ]", '', regex=True)
# 2.4 collapses multiple spaces
df['review'] = df['review'].str.replace(r'\s+', ' ', regex=True).str.strip()
# 2.5 trims the result
#print(df.head())
# Output: add a new column: clean_review
df['clean_review'] = df['review']
#print(df.head())

#Step 3 — Tokenization Using Model Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

user_model_dir = LOCAL_MODEL_DIR
if not (user_model_dir / 'config.json').exists():
    raise RuntimeError(f"Model config.json not found in: {user_model_dir}")

try:
    tokenizer = AutoTokenizer.from_pretrained(user_model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(user_model_dir, local_files_only=True)
except (requests.exceptions.SSLError, OSError, ValueError) as exc:
    raise RuntimeError(
        "Failed to load local sentiment model/tokenizer. Ensure provided directory contains files like "
        "config.json, model.safetensors (or pytorch_model.bin), and tokenizer files."
    ) from exc

df['tokens'] = df['clean_review'].apply(tokenizer.tokenize)

#print(df.head())

#Step 4 — Convert Tokens to Numerical IDs
#Output: new column: token_ids
df['token_ids'] = df['tokens'].apply(tokenizer.convert_tokens_to_ids)

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

def encode_for_model(text):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
    )
    return encoded['input_ids'], encoded['attention_mask']

df[['input_ids', 'attention_mask']] = df['clean_review'].apply(
    lambda text: pd.Series(encode_for_model(text))
)

#print(df.head())

#Step 6 — Run a Pretrained BERT Sentiment Classifier
'''
Use a local sentiment model.
For each review, run the model to obtain:

raw sentiment label
probability vector from softmax
Output: two columns:

sentiment_raw
probabilities
'''
id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else {}


def normalize_label(raw_label, predicted_idx):
    label = str(raw_label).strip().lower()
    if 'neg' in label:
        return 'negative'
    if 'neu' in label:
        return 'neutral'
    if 'pos' in label:
        return 'positive'

    # Fallback for generic labels such as LABEL_0, LABEL_1.
    if model.config.num_labels == 2:
        index_map = {0: 'negative', 1: 'positive'}
        return index_map.get(predicted_idx, 'negative')

    # Fallback for generic labels such as LABEL_0, LABEL_1, LABEL_2.
    if model.config.num_labels == 3:
        index_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return index_map.get(predicted_idx, 'neutral')

    return f"label_{predicted_idx}"

def predict_sentiment(input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        predicted_idx = int(torch.argmax(logits, dim=1).item())
        raw_label = id2label.get(predicted_idx, f'LABEL_{predicted_idx}')
        sentiment_raw = normalize_label(raw_label, predicted_idx)
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
df['final_sentiment'] = df.apply(
    lambda row: format_final_sentiment(row['sentiment_raw'], row['probabilities']),
    axis=1
)
# Save predictions.csv with original review and final sentiment prediction
df[['review', 'final_sentiment']].to_csv(PREDICTIONS_CSV, index=False)
# Save result.csv with all intermediate steps
df.to_csv(RESULT_CSV, index=False)

print(df.head())