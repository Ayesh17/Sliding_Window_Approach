import os
import pandas as pd

# Base directory and the behavior we are focusing on
behavior = 'headon'
input_behavior_dir = os.path.join("HMM_data", behavior, "scenario")
output_behavior_dir = os.path.join("HMM_data", "headon_preprocessed", "scenario")

# Ensure the output directory exists
os.makedirs(output_behavior_dir, exist_ok=True)

# Column headers for the files (assuming all files have 39 columns)
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

# List of all CSV files in the headon scenario folder
csv_files = [file for file in os.listdir(input_behavior_dir) if file.endswith('.csv')]

# Loop through each file in the 'headon' behavior folder
for file in csv_files:
    input_file_path = os.path.join(input_behavior_dir, file)

    # Read the CSV file without headers
    data = pd.read_csv(input_file_path, header=None)

    # Check if the number of columns matches the expected 39
    if data.shape[1] == len(column_headers):
        # Assign column names
        data.columns = column_headers

        # Remove frames where S_BEAM == 1
        data_filtered = data[data['S_BEAM'] != 1]

        # Save the filtered data to the new subfolder
        output_file_path = os.path.join(output_behavior_dir, file)
        data_filtered.to_csv(output_file_path, index=False, header=False)

        print(f"Processed file: {file} - Removed {len(data) - len(data_filtered)} frames with S_BEAM == 1")
    else:
        print(f"Skipping {file}: Expected 39 columns, found {data.shape[1]}")

print(f"\nAll files processed. Updated datasets saved in {output_behavior_dir}")
