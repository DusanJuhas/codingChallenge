"""
Step 1: Load the input CSV file.
"""

import pandas as pd


class DataLoader:
    """
    Responsible for loading the raw input CSV into a DataFrame.
    """

    def __init__(self, input_file: str):
        self.input_file = input_file

    def load(self) -> pd.DataFrame:
        """
        Load the CSV file specified by self.input_file.

        Raises:
            FileNotFoundError: if the file does not exist.
        """
        try:
            df = pd.read_csv(self.input_file)
            print("CSV file loaded successfully!")
        except FileNotFoundError as exc:
            print(f"Error: The file '{self.input_file}' was not found.")
            raise exc

        print("\nOriginal data preview:")
        print(df.head())
        return df
