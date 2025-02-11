import os
import glob

def rename_files_in_directory(input_dir):
    """
    Recursively renames all 'output_formatted.csv' files in the input directory
    and its subfolders to include the behavior (subfolder name) and a unique instance number.

    Example: 'output_formatted.csv' -> 'behavior_scenario_hmm_formatted.csv'
    """
    # Get top-level subfolders (behaviors)
    behavior_folders = [f for f in os.scandir(input_dir) if f.is_dir()]
    print(f"Found Behavior Subfolders: {[folder.name for folder in behavior_folders]}")

    for behavior_folder in behavior_folders:
        print(f"\n\n\nProcessing Behavior: {behavior_folder.name}")

        # Recursively find all subfolders within the behavior folder
        scenario_folders = [f for f in os.scandir(behavior_folder.path) if f.is_dir()]

        for scenario_folder in scenario_folders:
            print(f"  Processing Scenario Folder: {scenario_folder.name}")

            # Find all 'output_formatted.csv' files within the scenario folder
            csv_files = glob.glob(os.path.join(scenario_folder.path, 'output_hmm_formatted.csv'))
            print("csv_files", csv_files)

            for instance_number, csv_file in enumerate(csv_files, start=1):
                # Build the new filename with behavior, scenario, and a unique identifier
                parent_folder = os.path.dirname(csv_file)
                new_filename = f"{behavior_folder.name}_{scenario_folder.name}_hmm_formatted.csv"
                new_file_path = os.path.join(parent_folder, new_filename)

                # Rename the file
                os.rename(csv_file, new_file_path)
                print(f"    Renamed: {csv_file} -> {new_file_path}")

    print("\nFile Renaming Completed!")

if __name__ == '__main__':
    input_dir = os.path.join("../HMM_train_data_noiseless")  # Update with your input directory path
    rename_files_in_directory(input_dir)
