import os
import pandas as pd
import numpy as np

# Define input and output directories
input_root_folder = "../Datasets/Data_1000"
output_root_folder = "../Datasets/Data_1000_new"

os.makedirs(output_root_folder, exist_ok=True)
train_output_folder = os.path.join(output_root_folder, "train")
val_output_folder = os.path.join(output_root_folder, "validation")
test_output_folder = os.path.join(output_root_folder, "test")

os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# Function to process and pad CSV files
def process_and_pad_csv(file_path, output_folder, min_frames=150, target_frames=500):
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Ignore files with fewer than 150 rows
        if len(df) < min_frames:
            print(f"Skipping {file_path}: Less than {min_frames} rows.")
            return

        # Truncate files with more than 500 rows
        if len(df) > target_frames:
            df = df.iloc[:target_frames]
            print(f"Truncated {file_path} to {target_frames} rows.")

        # Pad files with fewer than 500 rows
        elif len(df) < target_frames:
            pad_len = target_frames - len(df)
            pad_df = pd.DataFrame(np.zeros((pad_len, len(df.columns))), columns=df.columns)
            pad_df['Label'] = -1  # Set labels of padded rows to -1
            df = pd.concat([df, pad_df], ignore_index=True)
            print(f"Padded {file_path} to {target_frames} rows (Label = -1 for padded rows).")

        # Save the processed file
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(output_file, index=False)
        print(f"Saved padded file: {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all files in train, validation, and test folders
for split in ["train", "validation", "test"]:
    input_folder = os.path.join(input_root_folder, split)
    output_folder = os.path.join(output_root_folder, split)

    if not os.path.exists(input_folder):
        print(f"Warning: {input_folder} does not exist. Skipping.")
        continue

    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing {split} dataset...")

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            process_and_pad_csv(file_path, output_folder)

print("Padding process completed successfully.")
