import os
import pandas as pd

# Define dataset root folder
dataset_root = "../Datasets/Binary_Data"  # Update this if needed
subfolders = ["train", "validation", "test"]  # All three datasets

# Function to remove the second occurrence of "S_drel_bear" (appears as "S_drel_bear.1")
def remove_duplicate_column(file_path):
    try:
        df = pd.read_csv(file_path)

        # Check if "S_drel_bear.1" exists
        if "S_drel_bear.1" in df.columns:
            print(f"Fixing duplicate 'S_drel_bear' in: {file_path}")

            # Drop the "S_drel_bear.1" column
            df = df.drop(columns=["S_drel_bear.1"])

            # Save the corrected file (overwrite)
            df.to_csv(file_path, index=False)
            print(f"✅ Fixed and saved: {file_path}")
        else:
            print(f"No duplicates found in: {file_path}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Process all CSV files in train, validation, and test directories
for subfolder in subfolders:
    folder_path = os.path.join(dataset_root, subfolder)

    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                remove_duplicate_column(file_path)
    else:
        print(f"⚠️ Folder not found: {folder_path}")

print("\n🎯 All dataset files processed. Duplicate 'S_drel_bear.1' columns removed where necessary.")
