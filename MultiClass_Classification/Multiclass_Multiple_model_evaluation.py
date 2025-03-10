import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Define Directory Paths
confusion_matrices_folder = f"../Results/Confusion_Matrices"
classification_reports_folder = f"../Results/Classification_Reports"
training_log_file = "../results/results_log.csv"  # File containing model names and test files

# Ensure required directories exist
os.makedirs(confusion_matrices_folder, exist_ok=True)
os.makedirs(classification_reports_folder, exist_ok=True)

# Behavior Label Mapping
behavior_mapping = {
    0: "benign",
    1: "block",
    2: "ram",
    3: "cross",
    4: "headon",
    5: "herd",
    6: "overtake"
}

# Load model names and dataset variants from the results log CSV file
if not os.path.exists(training_log_file):
    raise FileNotFoundError(f"Results log file {training_log_file} not found!")

training_log_df = pd.read_csv(training_log_file)

# Ensure required columns exist
required_columns = {"model_name", "dataset_variant"}
if not required_columns.issubset(training_log_df.columns):
    raise ValueError(f"Missing required columns in {training_log_file}. Required: {required_columns}")

# Get unique model types
model_types = training_log_df["model_name"].dropna().unique().tolist()

# Function to get dataset_variant dynamically
def get_dataset_variant(model_type):
    row = training_log_df.loc[training_log_df["model_name"] == model_type]
    if row.empty:
        raise ValueError(f"No dataset variant found for model {model_type}")
    return row["dataset_variant"].values[0]  # Extract the dataset_variant from the matched row


# Load Test Data with Sliding Windows
def load_data_from_files(file_list, window_size=20, step_size=5):
    sequences, labels = [], []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            if df.empty or 'Label' not in df.columns:
                print(f"Skipping {file_path}: File is empty or missing 'Label'. Columns found: {df.columns}")
                continue

            feature_columns = [col for col in df.columns if col != 'Label']
            full_sequence = df[feature_columns].values.astype(np.float32)

            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]
                if window.shape != (window_size, len(feature_columns)):
                    continue

                window_labels = df['Label'][start:start + window_size].values
                if (window_labels == -1).any():
                    continue

                majority_label = Counter(window_labels).most_common(1)[0][0]
                sequences.append(window)
                labels.append(majority_label)
        except Exception as e:
            print(f"Skipping {file_path}: Unexpected error - {e}. File may be corrupted or not formatted correctly.")

    if len(sequences) == 0:
        print(f"Warning: No valid sequences extracted from test files.")
    return np.array(sequences), np.array(labels)


# Initialize Results Storage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_type):
    dataset_variant = get_dataset_variant(model_type)  # Get dataset variant dynamically
    test_data_folder = f"../Datasets/{dataset_variant}/test"  # Update test data path
    print(f"Evaluating {model_type.upper()} using dataset {dataset_variant}")

    if not os.path.exists(test_data_folder):
        print(f"Test data directory {test_data_folder} not found. Skipping...")
        return None

    # Load test files
    test_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith(".csv")]
    test_files = [file for file in test_files if os.path.exists(file)]  # Filter missing files

    if len(test_files) == 0:
        print(f"No valid test files found for dataset {dataset_variant}. Skipping model {model_type}.")
        return None

    # Model path
    model_path = f"../Models/{model_type}"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping...")
        return None

    print(f"Evaluating {model_type.upper()}...")

    torch.cuda.empty_cache()
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Load Test Data
    X_test, y_test = load_data_from_files(test_files)
    if len(X_test) == 0:
        print(f"No valid sequences found. Check data format.")
        return None

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate Model
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Save results to the results log
    training_log_df.loc[training_log_df["model_name"] == model_type, "Accuracy"] = accuracy
    training_log_df.loc[training_log_df["model_name"] == model_type, "Precision"] = precision
    training_log_df.loc[training_log_df["model_name"] == model_type, "Recall"] = recall
    training_log_df.loc[training_log_df["model_name"] == model_type, "F1 Score"] = f1

    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=list(behavior_mapping.values()), zero_division=0)
    with open(os.path.join(classification_reports_folder, f"{model_type}_report.txt"), "w") as f:
        f.write(report)

    print(f"Results for {model_type.upper()} saved.")


# Run Evaluation for Each Model
for model in model_types:
    evaluate_model(model)

# Save updated results log
training_log_df.to_csv(training_log_file, index=False)
print(f"Updated results log saved to {training_log_file}")
