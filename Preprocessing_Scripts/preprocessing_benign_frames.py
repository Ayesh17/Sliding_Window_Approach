import os
import shutil
import pandas as pd

def process_csv_files(input_dir, output_dir):
    """
    Process all CSV files in subfolders to:
    1. Copy rows where 'beh_label' != 1 to the same subfolder structure in the output directory.
    2. Extract sequences of 20 or more continuous rows where 'beh_label' == 1 into a new 'benign_ext' folder.
    3. Copy the entire 'benign' folder without changes to the output directory.
    """
    # Directory for benign sequences
    benign_ext_dir = os.path.join(output_dir, "benign_ext")
    os.makedirs(benign_ext_dir, exist_ok=True)

    # Copy the benign folder without changes
    benign_folder = os.path.join(input_dir, "benign")
    if os.path.exists(benign_folder):
        shutil.copytree(benign_folder, os.path.join(output_dir, "benign"), dirs_exist_ok=True)
        print(f"Copied 'benign' folder to {output_dir}/benign")

    # Process other behavior folders
    behavior_folders = [f for f in os.scandir(input_dir) if f.is_dir() and f.name != "benign"]
    print(f"Processing Behavior Folders: {[folder.name for folder in behavior_folders]}")

    for behavior_folder in behavior_folders:
        print(f"Processing Behavior: {behavior_folder.name}")

        # Recursively process all CSV files in the behavior folder
        csv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(behavior_folder.path) for f in filenames if f.endswith('.csv')]

        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)

                # Check if 'beh_label' column exists
                if 'beh_label' not in df.columns:
                    print(f"Skipping {csv_file}: 'beh_label' column not found.")
                    continue

                # Identify benign rows and non-benign rows
                benign_rows = df[df['beh_label'] == 1]
                non_benign_rows = df[df['beh_label'] != 1]

                # Save continuous benign sequences of 20+ frames to a new file
                if not benign_rows.empty:
                    benign_sequences = extract_benign_sequences(benign_rows)
                    for seq_num, seq_df in enumerate(benign_sequences, start=1):
                        new_filename = f"benign_{os.path.basename(csv_file)}"
                        seq_file_path = os.path.join(benign_ext_dir, new_filename)
                        seq_df.to_csv(seq_file_path, index=False)
                        print(f"  Extracted benign sequence to {seq_file_path}")

                # Save non-benign rows to the new folder under the same subfolder structure
                if not non_benign_rows.empty:
                    relative_path = os.path.relpath(csv_file, input_dir)  # Preserve the subfolder structure
                    new_file_path = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                    non_benign_rows.to_csv(new_file_path, index=False)
                    print(f"  Copied non-benign frames to {new_file_path}")

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    print("\nProcessing Completed!")


def extract_benign_sequences(benign_rows):
    """
    Extract continuous sequences of 20+ frames where 'beh_label' == 1.
    """
    sequences = []
    current_sequence = []

    previous_index = None
    for index, row in benign_rows.iterrows():
        if previous_index is None or index == previous_index + 1:
            current_sequence.append(row)
        else:
            if len(current_sequence) >= 20:
                sequences.append(pd.DataFrame(current_sequence))
            current_sequence = [row]
        previous_index = index

    # Check the last sequence
    if len(current_sequence) >= 20:
        sequences.append(pd.DataFrame(current_sequence))

    return sequences


if __name__ == '__main__':
    input_dir = "../HMM_train_data_noise"  # Update with your input directory path
    output_dir = "../HMM_train_data_noise_preprocessed"  # Update with your output directory path
    os.makedirs(output_dir, exist_ok=True)
    process_csv_files(input_dir, output_dir)
