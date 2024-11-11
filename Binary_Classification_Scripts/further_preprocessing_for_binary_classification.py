import os
import pandas as pd

# Define the root directories for the existing dataset and the new binary dataset
input_root_folder = "Data"  # Folder containing the train, validation, and test data with original labels
output_root_folder = "Binary_Data"  # New folder for storing binary-labeled data

# Ensure the output root folder and its subdirectories (train, validation, test) exist
train_output_folder = os.path.join(output_root_folder, "train")
val_output_folder = os.path.join(output_root_folder, "validation")
test_output_folder = os.path.join(output_root_folder, "test")
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# Function to update label columns and save to the new directory
def process_and_save_binary_labels(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'Label' column exists
                if 'Label' not in df.columns:
                    print(f"Skipping {file_path}: 'Label' column not found.")
                    continue

                # Update the 'Label' column: 1 if label is 1, 2, or 5, else 0
                df['Label'] = df['Label'].apply(lambda x: 1 if x in [1, 2, 5] else 0)

                # Save the updated DataFrame to the new output folder
                output_file_path = os.path.join(output_folder, file)
                df.to_csv(output_file_path, index=False)
                print(f"Saved updated file to {output_file_path}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Process the 'train', 'validation', and 'test' folders separately and save to corresponding new folders
process_and_save_binary_labels(os.path.join(input_root_folder, "train"), train_output_folder)
process_and_save_binary_labels(os.path.join(input_root_folder, "validation"), val_output_folder)
process_and_save_binary_labels(os.path.join(input_root_folder, "test"), test_output_folder)

print("Processing and label modification completed. All files saved in 'Binary_Data' folder.")
