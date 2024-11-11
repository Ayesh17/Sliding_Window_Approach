import os
import shutil

# Define the root directory
root_dir = 'HMM_Data'
output_dir = 'data_preprocessed'

# List of behavior types (subdirectories)
behaviors = ['benign', 'block', 'ram', 'cross', 'headon', 'herd', 'overtake']

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

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

    # Sort the files by size in descending order and select the top 1000 longest files
    longest_files = sorted(csv_files, key=os.path.getsize, reverse=True)[:1000]

    # Define the destination directory path for the current behavior
    dest_behavior_dir = os.path.join(output_dir, behavior, 'scenario')
    os.makedirs(dest_behavior_dir, exist_ok=True)

    # Copy each selected file to the destination directory
    for file_path in longest_files:
        shutil.copy(file_path, dest_behavior_dir)
        print(f"Copied {file_path} to {dest_behavior_dir}")

print("Preprocessing complete. The longest 1000 files have been copied to 'data_preprocessed'.")
