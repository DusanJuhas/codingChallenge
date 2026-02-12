import pandas as pd

csv_file = "data/original.csv"
row_count = 10

try:
# task A.1 → Load the dataset
    df = pd.read_csv(csv_file)
    print(f"First {row_count} rows of the dataset:\n")
# task A.2.a → Display first 10 rows
    print(df.head(row_count))
# task A.2.b → Display the shape of the dataset
    print(df.shape)
# task A.2.c → Display the column names
    print(df.columns)
# task A.2.d → Display the data types by using the info() method
    print(df.info())

# task A.3.a → Display Missing values
    print("\nMissing values per column (count):")
    missing_counts = df.isna().sum()
    print(missing_counts)

    print("\nMissing values per column (percentage):")
    missing_pct = (missing_counts / len(df) * 100).round(2)
    print(missing_pct.astype(str) + " %")
  
# Rows that have ANY missing values
    rows_with_missing = df[df.isna().any(axis=1)]
    print(f"\nTotal rows with any missing value: {len(rows_with_missing)}")

    if len(rows_with_missing) > 0:
        print(f"\nShowing up to the rows with missing values: \n{rows_with_missing}")

# task A.3.b → Display Duplicated values
    duplicated_rows = df[df.duplicated()]
    print(f"\nTotal duplicated rows: {len(duplicated_rows)}")
    if len(duplicated_rows) > 0:
        print(f"\nShowing up to the duplicated rows: \n{duplicated_rows}")

except FileNotFoundError:
    print(f"Error: File not found at path '{csv_file}'.")
except Exception as e:
    print(f"An error occurred: {e}")
