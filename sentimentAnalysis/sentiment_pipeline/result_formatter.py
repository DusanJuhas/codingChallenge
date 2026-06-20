"""
Step 7: Produce final clean sentiment output and save results.
"""

import pandas as pd


class ResultFormatter:
    """
    Formats raw model output into a clean, human-readable sentiment
    string and handles saving results to disk.
    """

    @staticmethod
    def final_prediction(label, probabilities) -> str:
        """
        Format the final sentiment label with a confidence percentage.
        """
        positive_prob = probabilities[1]
        negative_prob = probabilities[0]
        confidence = round(max(positive_prob, negative_prob) * 100, 2)
        return f"{label} ({confidence}%)"

    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final_prediction() row-wise, producing the
        'final_sentiment' column.
        """
        df["final_sentiment"] = df.apply(
            lambda row: self.final_prediction(row["sentiment_raw"], row["probabilities"]),
            axis=1
        )
        print("\nFinal sentiment prediction preview:")
        print(df[["clean_review", "final_sentiment"]].head())
        return df

    @staticmethod
    def save(df: pd.DataFrame, predictions_file: str, full_output_file: str) -> None:
        """
        Save both the simplified human-readable predictions file
        and the full debug/log dataset.
        """
        # Save simplified predictions file
        df[["review", "final_sentiment"]].to_csv(predictions_file, index=False)
        print(f"\nHuman-readable predictions saved to {predictions_file}")

        # Save the full dataset for debugging
        df.to_csv(full_output_file, index=False)
        print(f"Full result saved to {full_output_file}")
