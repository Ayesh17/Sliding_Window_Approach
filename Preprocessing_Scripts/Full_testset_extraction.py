import os
import random
import shutil
from collections import defaultdict

# Set your folder paths
source_folder = "../Datasets/Data/test_old"
destination_folder = "../Datasets/Data/test"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get all files in the source folder
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Group files by behavior (part before the first '_')
behavior_files = defaultdict(list)
for file in all_files:
    behavior = file.split('_')[0]  # Extract behavior from filename
    behavior_files[behavior].append(file)

# Define number of files to select
total_files_needed = 1704
selected_files = []
selection_counts = {}

# Balance the selection as much as possible
num_behaviors = len(behavior_files)
files_per_behavior = total_files_needed // num_behaviors

# First, take an equal amount from each behavior (if possible)
for behavior, files in behavior_files.items():
    if len(files) >= files_per_behavior:
        selected = random.sample(files, files_per_behavior)
    else:
        selected = files  # If not enough files, take all available

    selected_files.extend(selected)
    selection_counts[behavior] = len(selected)

# If we didn't reach 1704 files, fill the remaining quota
remaining_files_needed = total_files_needed - len(selected_files)

if remaining_files_needed > 0:
    # Collect remaining files from behaviors that still have unselected files
    remaining_pool = [file for behavior, files in behavior_files.items() for file in files if
                      file not in selected_files]
    additional_selected = random.sample(remaining_pool, min(len(remaining_pool), remaining_files_needed))

    selected_files.extend(additional_selected)

    # Count the additional selections per behavior
    for file in additional_selected:
        behavior = file.split('_')[0]
        selection_counts[behavior] += 1

# Copy the selected files
for file in selected_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

# Print selection statistics
print(f"Successfully copied {len(selected_files)} files to {destination_folder}.")
print("File selection per behavior:")
for behavior, count in selection_counts.items():
    print(f"  {behavior}: {count} files")
