import os
import pandas as pd

# Define the directory containing training CSV files
train_data_folder = "../Datasets/Data15_No_Normalization/test"  # Update this if needed

# Set the expected number of features
expected_num_features = 15


# Function to check feature count in each CSV file
def check_feature_count(folder_path, expected_num_features):
    incorrect_files = []

    # Collect all CSV files in the folder
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if 'Label' column exists
            if 'Label' not in df.columns:
                print(f"{csv_file} is missing the 'Label' column.")
                incorrect_files.append(csv_file)
                continue

            # Count feature columns (excluding 'Label')
            feature_columns = [col for col in df.columns if col != 'Label']
            num_features = len(feature_columns)

            # Verify feature count
            if num_features != expected_num_features:
                print(f"{csv_file} has {num_features} features instead of {expected_num_features}.")
                incorrect_files.append(csv_file)

        except Exception as e:
            print(f"Could not process {csv_file}: {e}")
            incorrect_files.append(csv_file)

    if not incorrect_files:
        print("All files have the correct number of features.")
    else:
        print("Files with incorrect feature count:", incorrect_files)


# Run the check
check_feature_count(train_data_folder, expected_num_features)
