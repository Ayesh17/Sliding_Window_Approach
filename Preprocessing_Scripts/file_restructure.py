import os
import glob
import shutil

import os
print("Current Working Directory:", os.getcwd())

def main():
    # input_dir = os.path.join("../HMM_train_data_noise_preprocessed")
    # output_dir = os.path.join("../HMM_train_data")
    # input_dir = os.path.join("../HMM_test_data_noise")
    # output_dir = os.path.join("../HMM_test_data")
    input_dir = os.path.join("../DEC18OW")
    output_dir = os.path.join("../DEC180W_updated")
    restructure_folders(input_dir, output_dir)

def restructure_folders(input_dir, output_dir):
    """
    Restructure CSV files into a new folder hierarchy:
    - All files ending with 'hmm_formatted' are moved into a 'scenario' folder under each subfolder.
    - Sub-subfolders are ignored, and all relevant files are consolidated at the subfolder level.
    """
    # Ensure output directory is clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Get sub-folders from input directory
    subfolders = [f for f in os.scandir(input_dir) if f.is_dir()]
    print(f"Found Subfolders: {[folder.name for folder in subfolders]}")

    for subfolder in subfolders:
        print(f"Processing: {subfolder.name}")

        # Create 'scenario' folder inside the output subfolder
        scenario_folder_path = os.path.join(output_dir, subfolder.name, "scenario")
        os.makedirs(scenario_folder_path, exist_ok=True)

        # Find all CSV files ending with 'hmm_formatted.csv' within subfolder and its sub-subfolders
        csv_files = glob.glob(os.path.join(subfolder.path, '**', '*hmm_formatted.csv'), recursive=True)

        for csv_file in csv_files:
            destination_file = os.path.join(scenario_folder_path, os.path.basename(csv_file))
            shutil.copy(csv_file, destination_file)
            print(f"Copied: {csv_file} -> {destination_file}")

    print("\nFolder Restructure Completed!")

if __name__ == '__main__':
    main()
