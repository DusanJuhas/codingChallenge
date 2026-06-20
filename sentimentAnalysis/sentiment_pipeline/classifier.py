"""
Step 6: Real BERT Sentiment Classification (IMDB model).
"""

import torch
import pandas as pd
from transformers import BertForSequenceClassification


class SentimentClassifier:
    """
    Loads a pretrained BERT sentiment classifier and runs inference
    on encoded review text.
    """

    def __init__(self, model_name: str = "textattack/bert-base-uncased-imdb"):
        print("\nLoading BERT IMDB sentiment model...")
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def classify(self, input_ids, attention_mask):
        """
        Perform inference using the pretrained IMDB BERT classifier.
        Returns the raw sentiment label and probability vector.
        """
        ids_tensor = torch.tensor(input_ids).unsqueeze(0)
        mask_tensor = torch.tensor(attention_mask).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=ids_tensor, attention_mask=mask_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).tolist()[0]

        sentiment_idx = int(torch.argmax(logits, dim=1).item())
        sentiment_label = "positive" if sentiment_idx == 1 else "negative"

        return sentiment_label, probabilities

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply classify() row-wise across the DataFrame, producing
        'sentiment_raw' and 'probabilities' columns.
        """
        df["sentiment_raw"], df["probabilities"] = zip(
            *df.apply(
                lambda row: self.classify(row["input_ids"], row["attention_mask"]),
                axis=1,
            )
        )
        print("\nRaw sentiment prediction preview:")
        print(df[["clean_review", "sentiment_raw", "probabilities"]].head())
        return df
