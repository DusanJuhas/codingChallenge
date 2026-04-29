"""
===========================================================================================
Project for sentiment analysis of movie reviews

It will load CSV file movie_reviews.csv with column review, will clean the data,
then tokenize it and clasify the sentiment using BERT.
At the end it will create two output files:
       predictions.csv - file containing only the review and its sentiment
       result.csv      - file containg all of the partial steps during the analysis
===========================================================================================
"""

import torch
import pandas as pd
from transformers import BertTokenizer





FILE_NAME = "movie_reviews.csv"

def open_file(file_name):
    """Opens the given file and prints its head()"""
    try:
        df = pd.read_csv(file_name)
        print("CSV file loaded.")
        print(df.head())
    except FileNotFoundError as exc:
        print("File " + file_name + " was not found.")
        raise exc
    return df

def clean_text(data_frame):
    """Gets dataframe and in column "review" cleans the text data"""
    if "review" not in list(reviews.columns):
        print("Column review is not contained in file " + FILE_NAME)
    else:
        data_frame["clean_review"] = data_frame["review"]

        print("Lowercasing all characters.")
        data_frame["clean_review"] = data_frame["clean_review"].str.lower()

        print("Deleting HTML tags.")
        data_frame["clean_review"] = data_frame["clean_review"].str.replace(r"<.*?>", "", regex=True)

        print("Deleting nonalphanumeric characters.")
        data_frame["clean_review"] = data_frame["clean_review"].str.replace(r"[^A-Za-z0-9.,!? ]", "", regex=True)

        print("Deleting excesive spaces.")
        data_frame["clean_review"] = data_frame["clean_review"].str.replace(r"\s+", " ", regex=True)
        data_frame["clean_review"] = data_frame["clean_review"].str.strip()

    return data_frame

def tokenize_data(data_frame):
    """Gets data frame and tokenize the data from "clean_review" column."""
    list_of_tokenized = []
    for review_string in data_frame["clean_review"]:
        tokenized = tokenizer.tokenize(review_string)
        list_of_tokenized.append(tokenized)
    data_frame["tokens"] = list_of_tokenized
    return data_frame

def transform_to_ids(data_frame):
    """Gets data frame and transform the tokens to IDs"""
    list_of_ids = []
    for tokens in data_frame["tokens"]:
        ids = tokenizer.convert_tokens_to_ids(tokens)
        list_of_ids.append(ids)
    data_frame["token_ids"] = list_of_ids
    return data_frame

def padding_and_attention_mask(data_frame):
    """Getsdata frame with cleaned reviews and add there padded tokens and attention mask."""
    encoded = tokenizer(reviews["clean_review"].tolist(),padding = True, return_attention_mask = True)
    data_frame["input_ids"] = encoded["input_ids"]
    data_frame["attention_mask"] = encoded["attention_mask"]
    return data_frame

reviews = open_file(FILE_NAME)
reviews = clean_text(reviews)

#Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

reviews = tokenize_data(reviews)

reviews = transform_to_ids(reviews)

reviews = padding_and_attention_mask(reviews)

#with torch.no_grad():
#    outputs = model(reviews["input_ids"])

print(reviews)
