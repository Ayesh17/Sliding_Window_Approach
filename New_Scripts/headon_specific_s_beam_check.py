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

total_frames = 0
s_beam_frames = 0
files_with_s_beam = 0

# Loop through each file in the 'headon' behavior folder
for file in csv_files:
    file_path = os.path.join(behavior_dir, file)

    # Read the CSV file without headers
    data = pd.read_csv(file_path, header=None)

    # Check if the number of columns matches the expected 39
    if data.shape[1] == len(column_headers):
        # Assign column names
        data.columns = column_headers

        # Count the total number of frames and frames with S_BEAM == 1
        total_frames_in_file = len(data)
        s_beam_frames_in_file = (data['S_BEAM'] == 1).sum()

        # Update overall counts
        total_frames += total_frames_in_file
        s_beam_frames += s_beam_frames_in_file

        # Check if this file has any frame with S_BEAM == 1
        if s_beam_frames_in_file > 0:
            files_with_s_beam += 1

        # Print percentage of frames with S_BEAM == 1 for this file
        if total_frames_in_file > 0:
            percent_s_beam_in_file = (s_beam_frames_in_file / total_frames_in_file) * 100
            print(f"File: {file} - {percent_s_beam_in_file:.2f}% of frames have S_BEAM == 1")
    else:
        print(f"Skipping {file}: Expected 39 columns, found {data.shape[1]}")

# Calculate overall percentage of frames with S_BEAM == 1
if total_frames > 0:
    overall_percent_s_beam = (s_beam_frames / total_frames) * 100
    print(f"\nTotal number of files: {len(csv_files)}")
    print(f"Number of files with S_BEAM == 1: {files_with_s_beam}")
    print(f"Overall percentage of frames with S_BEAM == 1: {overall_percent_s_beam:.2f}%")
else:
    print("No frames found in the dataset.")
