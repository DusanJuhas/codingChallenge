"""
Steps 3-5: Tokenization, ID conversion, padding & attention masks.
"""

import pandas as pd
from transformers import BertTokenizer


class BertTextEncoder:
    """
    Wraps the BERT tokenizer and handles all text-to-tensor
    preparation steps: tokenizing, ID conversion, padding, and
    attention mask creation.
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_len: int = 64):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    # ---------------- Step 3 ----------------
    def tokenize(self, text: str):
        """
        Tokenize text into WordPiece tokens using the BERT tokenizer.
        """
        return self.tokenizer.tokenize(text)

    def tokenize_dataframe(self, df: pd.DataFrame, source_col: str = "clean_review",
                            target_col: str = "tokens") -> pd.DataFrame:
        """
        Apply tokenize() to an entire DataFrame column.
        """
        df[target_col] = df[source_col].apply(self.tokenize)
        print("\nTokenized data preview:")
        print(df[[source_col, target_col]].head())
        return df

    # ---------------- Step 4 ----------------
    def tokens_to_ids(self, tokens):
        """
        Convert WordPiece tokens into BERT vocabulary IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokens_to_ids_dataframe(self, df: pd.DataFrame, source_col: str = "tokens",
                                 target_col: str = "token_ids") -> pd.DataFrame:
        """
        Apply tokens_to_ids() to an entire DataFrame column.
        """
        df[target_col] = df[source_col].apply(self.tokens_to_ids)
        print("\nToken ID preview:")
        print(df[[source_col, target_col]].head())
        return df

    # ---------------- Step 5 ----------------
    def encode_with_padding(self, text: str):
        """
        Encode text using the BERT tokenizer, generating:
        - input_ids (padded)
        - attention_mask (1=token, 0=padding)
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        return input_ids, attention_mask

    def encode_dataframe(self, df: pd.DataFrame, source_col: str = "clean_review") -> pd.DataFrame:
        """
        Apply encode_with_padding() to an entire DataFrame column,
        producing the 'input_ids' and 'attention_mask' columns.
        """
        df["input_ids"], df["attention_mask"] = zip(
            *df[source_col].apply(self.encode_with_padding)
        )
        print("\nInput IDs and Attention Mask preview:")
        print(df[[source_col, "input_ids", "attention_mask"]].head())
        return df
