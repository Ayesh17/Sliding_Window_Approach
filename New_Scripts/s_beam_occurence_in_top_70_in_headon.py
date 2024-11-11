import os
import pandas as pd

# Base directory and the behavior we are focusing on
behavior = 'headon'
behavior_dir = os.path.join("HMM_data", behavior, "scenario")

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
csv_files = [file for file in os.listdir(behavior_dir) if file.endswith('.csv')]

files_with_s_beam_in_first_70 = 0

# Loop through each file in the 'headon' behavior folder
for file in csv_files:
    file_path = os.path.join(behavior_dir, file)

    # Read the CSV file without headers
    data = pd.read_csv(file_path, header=None)

    # Check if the number of columns matches the expected 39
    if data.shape[1] == len(column_headers):
        # Assign column names
        data.columns = column_headers

        # Calculate the index for 70% of the frames
        total_frames_in_file = len(data)
        seventy_percent_index = int(total_frames_in_file * 0.7)

        # Extract the first 70% of the frames
        first_70_percent_data = data.iloc[:seventy_percent_index]

        # Check if any frame in the first 70% has S_BEAM == 1
        if (first_70_percent_data['S_BEAM'] == 1).any():
            files_with_s_beam_in_first_70 += 1
            print(f"File: {file} contains a frame with S_BEAM == 1 in the first 70% of frames")
    else:
        print(f"Skipping {file}: Expected 39 columns, found {data.shape[1]}")

# Print the result
print(f"\nTotal number of files checked: {len(csv_files)}")
print(f"Number of files with S_BEAM == 1 in the first 70% of frames: {files_with_s_beam_in_first_70}")
