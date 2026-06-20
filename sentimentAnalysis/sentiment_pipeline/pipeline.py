"""
Pipeline Orchestrator: wires together all stage classes in order.
"""

import pandas as pd

from .data_loader import DataLoader
from .text_cleaner import TextCleaner
from .bert_encoder import BertTextEncoder
from .classifier import SentimentClassifier
from .result_formatter import ResultFormatter


class SentimentPipeline:
    """
    Orchestrates the full end-to-end sentiment analysis pipeline,
    wiring together all the individual stage classes in order.
    """

    def __init__(self,
                 input_file: str = "movie_reviews.csv",
                 predictions_file: str = "predictions.csv",
                 output_file: str = "result.csv",
                 max_len: int = 64):
        self.input_file = input_file
        self.predictions_file = predictions_file
        self.output_file = output_file

        self.loader = DataLoader(input_file)
        self.cleaner = TextCleaner()
        self.encoder = BertTextEncoder(max_len=max_len)
        self.classifier = SentimentClassifier()
        self.formatter = ResultFormatter()

    def run(self) -> pd.DataFrame:
        """
        Execute all pipeline steps in sequence and return the
        final DataFrame containing all intermediate and final columns.
        """
        df = self.loader.load()                       # Step 1
        df = self.cleaner.clean_dataframe(df)          # Step 2
        df = self.encoder.tokenize_dataframe(df)       # Step 3
        df = self.encoder.tokens_to_ids_dataframe(df)  # Step 4
        df = self.encoder.encode_dataframe(df)         # Step 5
        df = self.classifier.classify_dataframe(df)    # Step 6
        df = self.formatter.format_dataframe(df)       # Step 7

        self.formatter.save(df, self.predictions_file, self.output_file)
        return df
