import os
import numpy as np
import pandas as pd

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

# Define the selected columns to retain
selected_columns = [
    'DISTANCE', 'S_DELTA_DIST', 'RED_SPEED', 'abs_dheading', 'S_REG_r_b', 'S_REG_b_r', 'cpa_time',
    'cpa_dist', 'rel_r_b', 'S_BEAM', 'rel_b_r', 'BLUE_SPEED', 'r_accel', 'abs_b_r', 'abs_r_b',
    'RED_HEADING', 'S_DREL_BEAR', 'S_DABS_HEADING', 'BLUE_HEADING', 'S_SIDE', 'S_CPA_DELTA_DIST',
    'S_CPA_TIME', 'S_ACCEL'
]

# Output folders
output_root_folder = "../HMM_data/Data"
train_output_folder = os.path.join(output_root_folder, "train")
val_output_folder = os.path.join(output_root_folder, "validation")
test_output_folder = os.path.join(output_root_folder, "test")
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# Root folder of input datasets
root_folder = "../HMM_data"

# Behavior classes and label mapping
behavior_classes = ["benign", "block", "ram", "cross", "headon", "herd", "overtake"]
label_mapping = {"benign": 0, "block": 1, "ram": 2, "cross": 3, "headon": 4, "herd": 5, "overtake": 6}


# Function to preprocess without normalization
def preprocess(file_path, behavior_label):
    try:
        # Read the file without headers and assign column names
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] != len(column_headers):
            print(f"Skipping {file_path}: Incorrect column count.")
            return None

        # Assign headers and retain selected columns
        df.columns = column_headers
        df = df[selected_columns]

        # Remove files with fewer than 200 rows
        if len(df) < 200:
            print(f"Skipping {file_path}: Less than 200 frames.")
            return None

        # Add label column
        df['Label'] = behavior_label

        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Process and save sequences
for split in ["train", "validation", "test"]:
    for behavior in behavior_classes:
        folder_path = os.path.join(root_folder, split, behavior, "scenario")
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping.")
            continue

        print(f"Processing {split}/{behavior}...")
        behavior_label = label_mapping[behavior]
        sequence_index = 0

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                sequence_df = preprocess(file_path, behavior_label)

                if sequence_df is not None:
                    output_folder = {
                        "train": train_output_folder,
                        "validation": val_output_folder,
                        "test": test_output_folder
                    }[split]

                    # Save the file
                    output_file = os.path.join(output_folder, f"{behavior}_{sequence_index}.csv")
                    sequence_df.to_csv(output_file, index=False)
                    print(f"Saved {output_file}")
                    sequence_index += 1

print("Preprocessing completed successfully.")
