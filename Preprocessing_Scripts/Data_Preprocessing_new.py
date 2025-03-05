import os
import pandas as pd

# Define the column headers as provided
column_headers = [
    'time (ms)', 'BID', 'b_lat (deg)', 'b_lon (deg)', 'b_head (deg)', 'b_sp (mps)',
    'RID', 'r_lat (deg)', 'r_lon (deg)', 'r_head (deg)', 'r_speed (mps)',
    'dist (m)', 'beh_phase', 'colregs', 'avoidance', 'beh_label',
    'abs_b_r', 'abs_r_b', 'rel_b_r', 'rel_r_b', 'abs_dheading',
    'r_accel', 'dvx', 'dvy', 'xr', 'yr', 'cpa_time (s)', 'cpa_dist (m)',
    'S_side', 'S_beam', 'S_reg_b_r', 'S_reg_r_b', 'S_drel_bear', 'S_ddist',
    'S_accel', 'S_dabs_heading', 'S_cpa_t', 'S_cpa_ddist', 'S_cpa_tdist'
]

# Define the selected columns to retain
selected_columns = [
    'dist (m)', 'S_drel_bear', 'r_speed (mps)', 'abs_dheading', 'S_reg_r_b', 'S_reg_b_r', 'cpa_time (s)',
    'cpa_dist (m)', 'rel_r_b', 'S_beam', 'rel_b_r', 'b_sp (mps)', 'r_accel', 'abs_b_r', 'abs_r_b',
    'r_head (deg)', 'S_drel_bear', 'S_dabs_heading', 'b_head (deg)', 'S_side', 'S_cpa_ddist',
    'S_cpa_t', 'S_accel', 'beh_phase'
]


# Mapping HII_ID values to new numeric labels
hii_id_to_label = {
    6: 2,  # RAM -> 2
    1: 0,  # BENIGN -> 0
    8: 1,  # BLOCK -> 1
    5: 3,  # CROSS -> 3
    7: 5,  # HERD -> 5
    4: 6,  # OVERTAKE -> 6
    3: 4   # HEADON -> 4
}

# Ensure output directories exist
output_root_folder = "../Datasets/Data"
os.makedirs(output_root_folder, exist_ok=True)
train_output_folder = os.path.join(output_root_folder, "train")
val_output_folder = os.path.join(output_root_folder, "validation")
test_output_folder = os.path.join(output_root_folder, "test")
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# Path to root folder containing train, validation, and test subfolders
root_folder = "../HMM_data"

# Function to preprocess and filter files with < 200 rows
def preprocess_and_filter(file_path):
    try:
        # Read the CSV file normally (keeping existing headers)
        df = pd.read_csv(file_path)

        print(f"Read CSV file: {file_path} with shape: {df.shape}")

        # Check if the DataFrame has fewer than 200 rows
        if len(df) < 150:
            print(f"File {file_path} has less than 200 rows. Skipping.")
            return None

        # Ensure 'BEH_LABEL' exists before processing
        if 'beh_label' in df.columns:
            df = df[selected_columns + ['beh_label']]
            df = df.rename(columns={'beh_label': 'Label'})

            # Replace HII_ID values in the 'Label' column with the new numeric labels
            df['Label'] = df['Label'].replace(hii_id_to_label)
        else:
            print(f"BEH_LABEL column missing in file: {file_path}. Skipping.")
            return None

        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Label mapping dictionary for behavior types
behavior_classes = ["benign", "block", "ram", "cross", "headon", "herd", "overtake"]

# Iterate over each split ('train', 'validation', and 'test') and behavior
for split in ["train", "validation", "test"]:
    for behavior in behavior_classes:
        folder_path = os.path.join(root_folder, split, behavior, "scenario")
        if not os.path.exists(folder_path):
            print(f"Warning: The folder '{folder_path}' does not exist. Skipping this behavior in {split}.")
            continue

        print(f"Processing {split} behavior: {behavior}")
        sequence_index = 0  # Track sequence numbers for file naming

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                print(f"Reading file: {file_path}")

                # Process the file and filter it
                sequence_df = preprocess_and_filter(file_path)
                if sequence_df is not None:
                    # Determine the correct output subfolder based on split
                    if split == "train":
                        output_folder = train_output_folder
                    elif split == "validation":
                        output_folder = val_output_folder
                    else:
                        output_folder = test_output_folder

                    # Save each sequence as a CSV file
                    file_name = f"{output_folder}/{behavior}_{sequence_index}.csv"
                    sequence_df.to_csv(file_name, index=False)
                    print(f"Saved sequence {sequence_index} to {file_name}")
                    sequence_index += 1

print("Preprocessing completed. Files with < 200 rows were removed.")
