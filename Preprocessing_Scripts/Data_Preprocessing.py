import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the original column headers from the CSV files
column_headers = [
    'TIME', 'BLUE_ID', 'BLUE_LAT', 'BLUE_LON', 'BLUE_HEADING', 'BLUE_SPEED',
    'RED_ID', 'RED_LAT', 'RED_LON', 'RED_HEADING', 'RED_SPEED',
    'DISTANCE', 'BEH_PHASE', 'COLREGS', 'AVOIDANCE', 'BEH_LABEL',
    'abs_b_r', 'abs_r_b', 'rel_b_r', 'rel_r_b', 'abs_dheading',
    'r_accel', 'dvx', 'dvy', 'xr', 'yr', 'cpa_time', 'cpa_dist',
    'S_SIDE', 'S_BEAM', 'S_REG_b_r', 'S_REG_r_b', 'S_DREL_BEAR',
    'S_DELTA_DIST', 'S_ACCEL', 'S_DABS_HEADING', 'S_CPA_TIME',
    'S_CPA_DELTA_DIST', 'S_CPA_TIME_DIST'
]
#
# Define the 23 selected columns you want to retain
selected_columns = [
    'DISTANCE', 'S_DELTA_DIST', 'RED_SPEED', 'abs_dheading', 'S_REG_r_b', 'S_REG_b_r', 'cpa_time',
    'cpa_dist', 'rel_r_b', 'S_BEAM', 'rel_b_r', 'BLUE_SPEED', 'r_accel', 'abs_b_r', 'abs_r_b',
    'RED_HEADING', 'S_DREL_BEAR', 'S_DABS_HEADING', 'BLUE_HEADING', 'S_SIDE', 'S_CPA_DELTA_DIST',
    'S_CPA_TIME', 'S_ACCEL'
]


# # Define the 15 selected columns you want to retain
# selected_columns = [
#     'DISTANCE', 'S_DELTA_DIST', 'RED_SPEED', 'abs_dheading', 'S_REG_r_b', 'S_REG_b_r', 'cpa_time',
#     'cpa_dist', 'rel_r_b', 'S_BEAM', 'rel_b_r', 'BLUE_SPEED', 'r_accel', 'abs_b_r', 'abs_r_b'
# ]
#

# List of features to normalize
features_to_normalize = selected_columns

# Create new names for the normalized features
normalized_feature_names = [f"{col}_normalized" for col in features_to_normalize]

# Create final column names with new normalized names
final_columns = [f"{col}_normalized" for col in selected_columns] + ['Label']

# Ensure the root output directory exists
output_root_folder = "../Datasets/Data"
# output_root_folder = "Data15"
os.makedirs(output_root_folder, exist_ok=True)

# Ensure separate subdirectories for train, validation, and test within the output folder
train_output_folder = os.path.join(output_root_folder, "train")
val_output_folder = os.path.join(output_root_folder, "validation")
test_output_folder = os.path.join(output_root_folder, "test")
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# Path to the root folder containing the train, validation, and test subfolders
root_folder = "../Datasets/data_preprocessed"

# Set the maximum sequence length for padding/truncation
max_sequence_length = 500

# Function to preprocess, normalize selected features, and assign updated column headers
def preprocess_and_normalize(file_path):
    try:
        # Read the CSV file without headers and assign the custom column headers
        df = pd.read_csv(file_path, header=None)  # Read without headers
        print(f"Read CSV file: {file_path} with shape: {df.shape}")

        # Check if the number of columns matches the original 39 columns
        if df.shape[1] != len(column_headers):
            print(f"Column mismatch in file: {file_path}. Expected {len(column_headers)} columns, but got {df.shape[1]}. Skipping this file.")
            return None

        # Assign the original headers to the DataFrame
        df.columns = column_headers

        # Remove unwanted columns and retain only the selected 23 features
        df = df[selected_columns]

        # Check if the DataFrame is empty after column selection
        if df.empty:
            print(f"Warning: {file_path} is empty after selecting columns and will be skipped.")
            return None

        # Normalize the selected features and update column names
        scaler = MinMaxScaler()
        df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

        # Assign the new column names for normalized features
        df.columns = final_columns[:-1]  # Exclude 'Label' for now

        # Add a placeholder 'Label' column at the end
        df['Label'] = 0  # Replace 0 with appropriate labels if needed

        # Adjust the DataFrame to ensure it has exactly 500 rows
        if len(df) < max_sequence_length:
            # Pad with zeros if the sequence is shorter than 500 rows
            padding_df = pd.DataFrame(0, index=np.arange(max_sequence_length - len(df)), columns=df.columns)
            df = pd.concat([df, padding_df], ignore_index=True)
            print(f"Padded sequence in {file_path} to {max_sequence_length} rows.")
        elif len(df) > max_sequence_length:
            # Truncate if the sequence is longer than 500 rows
            df = df.iloc[:max_sequence_length]
            print(f"Truncated sequence in {file_path} to {max_sequence_length} rows.")

        # Convert the DataFrame to a NumPy array
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Read and preprocess all CSV files, and collect them into sequences with labels
data_sequences = []
labels = []
behavior_classes = ["benign", "block", "ram", "cross", "headon", "herd", "overtake"]

# Define label mapping dictionary for label transformation
label_mapping = {"benign": 0, "block": 1, "ram": 2, "cross": 3, "headon": 4, "herd": 5, "overtake": 6}

# Iterate over each split ('train', 'validation', and 'test') and behavior
for split in ["train", "validation", "test"]:
    for behavior in behavior_classes:
        folder_path = os.path.join(root_folder, split, behavior, "scenario")  # Include "scenario" subfolder in path
        if not os.path.exists(folder_path):
            print(f"Warning: The folder '{folder_path}' does not exist. Skipping this behavior in {split}.")
            continue

        print(f"Processing {split} behavior: {behavior}")
        behavior_label = label_mapping[behavior]
        sequence_index = 0  # Track sequence numbers for file naming

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                print(f"Reading file: {file_path}")

                # Process the file and normalize it
                sequence_df = preprocess_and_normalize(file_path)
                if sequence_df is not None:
                    # Assign the label based on the behavior type
                    sequence_df['Label'] = behavior_label

                    data_sequences.append(sequence_df.values)
                    labels.append(sequence_df['Label'].iloc[0])  # Use the label from the 'Label' column

                    # Determine the correct output subfolder based on split
                    if split == "train":
                        output_folder = train_output_folder
                    elif split == "validation":
                        output_folder = val_output_folder
                    else:
                        output_folder = test_output_folder

                    # Save each sequence as a CSV file with the specified naming convention and 'Label' column included
                    file_name = f"{output_folder}/{behavior}_{sequence_index}.csv"
                    sequence_df.to_csv(file_name, index=False)
                    print(f"Saved sequence {sequence_index} to {file_name}")
                    sequence_index += 1

print("Preprocessing completed. All sequences have been adjusted to the required length.")
