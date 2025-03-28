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
confusion_matrices_folder = "../Results/Confusion_Matrices"
classification_reports_folder = "../Results/Classification_Reports"
training_log_file = "../results/results_log_binary.csv"  # File containing model names and test files

# Ensure required directories exist
os.makedirs(confusion_matrices_folder, exist_ok=True)
os.makedirs(classification_reports_folder, exist_ok=True)

# Behavior Label Mapping for Binary Classification
behavior_mapping = {
    0: "benign",
    1: "hostile",
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


# Function to generate and save the confusion matrix
def save_confusion_matrix(y_true, y_pred, model_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=behavior_mapping.values(),
                yticklabels=behavior_mapping.values())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_type}")

    # Save the confusion matrix image
    cm_path = os.path.join(confusion_matrices_folder, f"{model_type}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved at {cm_path}")


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

            # Adjust prediction extraction for binary classification
            if outputs.shape[1] == 1:
                predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            else:
                predicted = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save confusion matrix
    save_confusion_matrix(all_labels, all_preds, model_type)

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    print("Recall per class:", recall_per_class)

    # Save classification report
    report = classification_report(all_labels, all_preds, labels=[0, 1], target_names=["benign", "hostile"], zero_division=0)
    report_path = os.path.join(classification_reports_folder, f"{model_type}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Results for {model_type.upper()} saved.")

    # ✅ Ensure that metrics columns exist in DataFrame
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        if metric not in training_log_df.columns:
            training_log_df[metric] = np.nan  # Create the column if missing

    # ✅ Update DataFrame with evaluation results
    training_log_df.loc[training_log_df["model_name"] == model_type, "Accuracy"] = accuracy
    training_log_df.loc[training_log_df["model_name"] == model_type, "Precision"] = precision
    training_log_df.loc[training_log_df["model_name"] == model_type, "Recall"] = recall
    training_log_df.loc[training_log_df["model_name"] == model_type, "F1 Score"] = f1

# ✅ Save the updated results log **AFTER** all models are evaluated
for model in model_types:
    evaluate_model(model)

training_log_df.to_csv(training_log_file, index=False)
print(f"Updated results log saved to {training_log_file}")
