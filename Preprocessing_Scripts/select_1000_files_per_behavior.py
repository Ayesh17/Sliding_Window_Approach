import os
import shutil
import csv

# Define the root directory
root_dir = '../HMM_train_data'
output_dir = '../HMM_train_data_preprocessed'

# root_dir = '../HMM_test_data'
# output_dir = '../HMM_test_data_preprocessed'

# List of behavior types (subdirectories)
behaviors = ['benign', 'block', 'ram', 'cross', 'headon', 'herd', 'overtake']

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

def count_csv_rows(file_path):
    """Counts the number of rows in a CSV file."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count

# Iterate through each behavior folder
for behavior in behaviors:
    # Define the path for the current behavior's scenario folder
    scenario_dir = os.path.join(root_dir, behavior, 'scenario')

    # Check if the directory exists
    if not os.path.exists(scenario_dir):
        print(f"Directory {scenario_dir} does not exist. Skipping...")
        continue

    # Get a list of all CSV files in the scenario directory
    csv_files = [os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir) if f.endswith('.csv')]

    # Filter files with more than 150 rows
    csv_files = [f for f in csv_files if count_csv_rows(f) > 150]

    # Sort the files by size in descending order and select the top 850 longest files (and select 150 more from test)
    longest_files = sorted(csv_files, key=os.path.getsize, reverse=True)[:1500]

    # Define the destination directory path for the current behavior
    dest_behavior_dir = os.path.join(output_dir, behavior, 'scenario')
    os.makedirs(dest_behavior_dir, exist_ok=True)

    # Copy each selected file to the destination directory
    for file_path in longest_files:
        shutil.copy(file_path, dest_behavior_dir)
        print(f"Copied {file_path} to {dest_behavior_dir}")

print("Preprocessing complete. The longest 1000 files with more than 200 rows have been copied to 'HMM_train_data_preprocessed'.")
