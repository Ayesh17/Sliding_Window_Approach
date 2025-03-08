import os
import pandas as pd

# Define source and destination directories
# source_root = "../Datasets/Data_hyperparam"
# destination_root = "../Datasets/Binary_Data_hyperparam"

# source_root = "../Datasets/Data_1000"
# destination_root = "../Datasets/Binary_Data_1000"

source_root = "../Datasets/Data_1000"
destination_root = "../Datasets/Binary_Data_1000"

# Define the correct mapping from preprocessed labels to binary labels
preprocessed_label_to_binary = {
    2: 1,  # RAM -> 1
    0: 0,  # BENIGN -> 0
    1: 1,  # BLOCK -> 1
    3: 0,  # CROSS -> 0
    5: 1,  # HERD -> 1
    6: 0,  # OVERTAKE -> 0
    4: 0   # HEADON -> 0
}

# Function to process and copy files
def process_and_copy_files(source_folder, destination_folder):
    """
    Recursively copies files while maintaining directory structure.
    Updates the 'Label' column in CSV files based on binary classification.
    """
    for root, dirs, files in os.walk(source_folder):
        # Compute the relative path from the source root
        relative_path = os.path.relpath(root, source_root)
        target_dir = os.path.join(destination_folder, relative_path)

        # Ensure the corresponding destination directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Process each CSV file in the directory
        for file in files:
            if file.endswith(".csv"):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(target_dir, file)

                try:
                    # Read the CSV file
                    df = pd.read_csv(source_file_path)

                    # Ensure 'Label' column exists before mapping
                    if 'Label' in df.columns:
                        # Convert to integer to avoid silent type mismatches
                        df['Label'] = df['Label'].astype(int)

                        # Apply the new binary classification mapping
                        df['Label'] = df['Label'].map(preprocessed_label_to_binary)

                        # Save the modified file in the new destination
                        df.to_csv(destination_file_path, index=False)

                        # Debugging: Print counts of 0s and 1s after transformation
                        label_counts = df['Label'].value_counts().to_dict()
                        print(f"Processed {file}: {label_counts}")

                    else:
                        print(f"Skipping {file} in {relative_path} (no 'Label' column)")

                except Exception as e:
                    print(f"Error processing {source_file_path}: {e}")

# Execute the processing function
process_and_copy_files(source_root, destination_root)

print("Binary dataset creation completed. Folder structure replicated, and labels updated.")
