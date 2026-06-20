"""
Step 2: Clean and preprocess the text.
"""

import re
import pandas as pd


class TextCleaner:
    """
    Responsible for cleaning raw review text before tokenization.
    """

    @staticmethod
    def clean(text: str) -> str:
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

    def clean_dataframe(self, df: pd.DataFrame, source_col: str = "review",
                         target_col: str = "clean_review") -> pd.DataFrame:
        """
        Apply clean() to an entire DataFrame column and store the
        result in a new column.
        """
        df[target_col] = df[source_col].apply(self.clean)
        print("\nCleaned data preview:")
        print(df.head())
        return df
