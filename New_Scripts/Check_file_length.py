import os
import pandas as pd

# Define the directory containing the CSV files
data_folder = "Data/train"
# data_folder = "Data/test"

# Collect all CSV files in the data folder (ignoring 'all_sequences_normalized.csv' for now)
csv_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".csv") and "all_sequences_normalized" not in file]

# Print the list of CSV files being processed
print(f"csv_files {csv_files}")

# Expected rows and columns for each file
expected_rows = 500
expected_columns = 24

# Iterate through each CSV file and print the number of rows and columns
for csv_file in csv_files:
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Get the number of rows and columns
        rows, columns = df.shape

        # Check if rows and columns match the expected values
        if rows != expected_rows or columns != expected_columns:
            raise ValueError(f"File {csv_file} has {rows} rows and {columns} columns, expected {expected_rows} rows and {expected_columns} columns.")

        # Print the filename and its dimensions if it meets the expected dimensions
        # print(f"{csv_file}: {rows} rows, {columns} columns")

    except ValueError as ve:
        # Print the exception message for mismatched rows or columns
        print(f"Dimension Error: {ve}")
    except pd.errors.EmptyDataError:
        print(f"{csv_file}: No data in the file (EmptyDataError).")
    except pd.errors.ParserError:
        print(f"{csv_file}: Parsing error (ParserError).")
    except Exception as e:
        print(f"{csv_file}: Unexpected error - {e}")
