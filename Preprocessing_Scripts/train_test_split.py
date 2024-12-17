import os
import shutil
import random

# Define the root directory for the preprocessed data
root_dir = '../Datasets/data_preprocessed2'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'validation')
test_dir = os.path.join(root_dir, 'test')

# List of behavior types (subdirectories)
behaviors = ['benign', 'block', 'ram', 'cross', 'headon', 'herd', 'overtake']

# Create the train, validation, and test directories inside the data_preprocessed directory if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate through each behavior folder
for behavior in behaviors:
    # Define the path for the current behavior's scenario folder in the preprocessed data directory
    scenario_dir = os.path.join(root_dir, behavior, 'scenario')

    # Check if the directory exists
    if not os.path.exists(scenario_dir):
        print(f"Directory {scenario_dir} does not exist. Skipping...")
        continue

    # Get a list of all CSV files in the scenario directory
    csv_files = [os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir) if f.endswith('.csv')]

    # Calculate the number of files for training, validation, and testing (70%, 15%, 15%)
    train_count = int(0.7 * len(csv_files))
    val_count = int(0.15 * len(csv_files))
    test_count = len(csv_files) - train_count - val_count

    # Check if there are enough files to split into train, validation, and test sets
    if len(csv_files) < train_count + val_count + test_count:
        print(f"Not enough files in {scenario_dir} to split into train, validation, and test sets. Skipping...")
        continue

    # Shuffle the files randomly
    random.shuffle(csv_files)

    # Split into training, validation, and testing sets
    train_files = csv_files[:train_count]
    val_files = csv_files[train_count:train_count + val_count]
    test_files = csv_files[train_count + val_count:]

    # Define destination directories for training, validation, and testing for the current behavior
    dest_train_behavior_dir = os.path.join(train_dir, behavior, 'scenario')
    dest_val_behavior_dir = os.path.join(val_dir, behavior, 'scenario')
    dest_test_behavior_dir = os.path.join(test_dir, behavior, 'scenario')
    os.makedirs(dest_train_behavior_dir, exist_ok=True)
    os.makedirs(dest_val_behavior_dir, exist_ok=True)
    os.makedirs(dest_test_behavior_dir, exist_ok=True)

    # Copy training files to the destination directory
    for file_path in train_files:
        shutil.copy(file_path, dest_train_behavior_dir)
        print(f"Copied {file_path} to {dest_train_behavior_dir}")

    # Copy validation files to the destination directory
    for file_path in val_files:
        shutil.copy(file_path, dest_val_behavior_dir)
        print(f"Copied {file_path} to {dest_val_behavior_dir}")

    # Copy testing files to the destination directory
    for file_path in test_files:
        shutil.copy(file_path, dest_test_behavior_dir)
        print(f"Copied {file_path} to {dest_test_behavior_dir}")

print("Data splitting complete. Training, validation, and testing files have been copied to 'data_preprocessed/train', 'data_preprocessed/validation', and 'data_preprocessed/test' directories respectively.")
