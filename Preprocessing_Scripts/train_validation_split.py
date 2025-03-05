import os
import shutil
import random

# Define the root directory for the preprocessed data
root_dir = '../HMM_train_data'
# root_dir = '../HMM_train_data_preprocessed'
# root_dir = '../HMM_train_data_preprocessed/hyperparam_tuning'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'validation')

# List of behavior types (subdirectories)
behaviors = ['benign', 'block', 'ram', 'cross', 'headon', 'herd', 'overtake']

# Create the train and validation directories inside the root directory if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

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

    # Calculate the number of files for training and validation (80%, 20%)
    train_count = int(0.8 * len(csv_files))
    val_count = len(csv_files) - train_count
    # train_count = 170
    # val_count = 30


    # Check if there are enough files to split into train and validation sets
    if len(csv_files) < train_count + val_count:
        print(f"Not enough files in {scenario_dir} to split into train and validation sets. Skipping...")
        continue

    # Shuffle the files randomly
    random.shuffle(csv_files)

    # Split into training and validation sets
    train_files = csv_files[:train_count]
    val_files = csv_files[train_count:train_count+val_count]

    # Define destination directories for training and validation for the current behavior
    dest_train_behavior_dir = os.path.join(train_dir, behavior, 'scenario')
    dest_val_behavior_dir = os.path.join(val_dir, behavior, 'scenario')
    os.makedirs(dest_train_behavior_dir, exist_ok=True)
    os.makedirs(dest_val_behavior_dir, exist_ok=True)

    # Copy training files to the destination directory
    for file_path in train_files:
        shutil.copy(file_path, dest_train_behavior_dir)
        print(f"Copied {file_path} to {dest_train_behavior_dir}")

    # Copy validation files to the destination directory
    for file_path in val_files:
        shutil.copy(file_path, dest_val_behavior_dir)
        print(f"Copied {file_path} to {dest_val_behavior_dir}")

    # Remove the original behavior folder
    shutil.rmtree(os.path.join(root_dir, behavior))
    print(f"Removed original folder: {os.path.join(root_dir, behavior)}")

print("Data splitting complete. Training and validation files have been copied to 'train' and 'validation' directories respectively.")
