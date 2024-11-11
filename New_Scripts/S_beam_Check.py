import os
import pandas as pd

# Base directory of the dataset
base_dir = "HMM_data"

# List of behaviors (subfolders)
behaviors = ['benign', 'block', 'ram', 'cross', 'headon', 'herd', 'overtake']

# Dictionary to store results
results = {}

for behavior in behaviors:
    behavior_dir = os.path.join(base_dir, behavior, "scenario")
    files = os.listdir(behavior_dir)

    # Filter out only .csv files
    csv_files = [file for file in files if file.endswith('.csv')]

    count = 0
    first_file_with_s_beam = None

    # Loop through each CSV file in the behavior folder
    for file in csv_files:
        print("file", file)
        file_path = os.path.join(behavior_dir, file)

        # Read the CSV file without headers, since column headers are not printed
        data = pd.read_csv(file_path, header=None)

        # Assign column names to the dataframe
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
        data.columns = column_headers

        # Check if any row has S_BEAM == 1
        if (data['S_BEAM'] == 1).any():
            count += 1
            if first_file_with_s_beam is None:
                first_file_with_s_beam = file

    # Store the count and the first file with S_BEAM == 1 for this behavior
    results[behavior] = {
        "count": count,
        "first_file": first_file_with_s_beam
    }

# Print the results
for behavior, result in results.items():
    print(f"Behavior: {behavior}")
    print(f"  Files with S_BEAM == 1: {result['count']}")
    if result['first_file']:
        print(f"  First file with S_BEAM == 1: {result['first_file']}")
    else:
        print("  No file with S_BEAM == 1")
