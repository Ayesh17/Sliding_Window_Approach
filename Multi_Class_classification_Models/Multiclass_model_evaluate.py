import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib  # For dynamic imports

# === Choose Model Type: "lstm" or "transformer" ===
model_type = "lstm"  # Change to "transformer" to switch models

# === Define Dataset Variant ===
dataset_variant = "Data"

# === Define Directory Paths ===
test_data_folder = f"../Datasets/{dataset_variant}/test"
confusion_matrices_folder = f"../Results/Confusion_Matrices"
classification_reports_folder = f"../Results/Classification_Reports"
os.makedirs(confusion_matrices_folder, exist_ok=True)
os.makedirs(classification_reports_folder, exist_ok=True)

# === Behavior Label Mapping ===
behavior_mapping = {
    0: "benign",
    1: "block",
    2: "ram",
    3: "cross",
    4: "headon",
    5: "herd",
    6: "overtake"
}

# === Detect the Number of Features ===
def get_feature_count(folder_path):
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the test data folder.")

    df = pd.read_csv(csv_files[0])
    feature_columns = [col for col in df.columns if col != 'Label']
    return len(feature_columns)

no_of_features = get_feature_count(test_data_folder)
print(f"Detected number of features: {no_of_features}")

# === Dynamically Import the Model ===
if model_type == "lstm":
    model_module = importlib.import_module("Multiclass_LSTM_model")
    ModelClass = model_module.LSTMClassifier
    model_path = f"../Models/lstm_model_{dataset_variant}.pth"
    hidden_size, num_layers = 128, 2  # Match training values
    model = ModelClass(input_size=no_of_features, hidden_size=hidden_size, num_layers=num_layers, num_classes=len(behavior_mapping))

elif model_type == "transformer":
    model_module = importlib.import_module("Multiclass_Transformer_model")
    ModelClass = model_module.TransformerClassifier
    model_path = f"../Models/transformer_model_{dataset_variant}.pth"
    d_model, num_heads, num_encoder_layers, dim_feedforward, dropout = 4, 2, 2, 128, 0.1
    model = ModelClass(input_size=no_of_features, num_classes=len(behavior_mapping), d_model=d_model,
                       nhead=num_heads, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
else:
    raise ValueError("Invalid model type. Choose 'lstm' or 'transformer'.")

print("Model path:", model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load the Model ===
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print(f"{model_type.upper()} Model loaded successfully.")

# === Load Test Data with Sliding Windows ===
def load_data_from_folder(folder_path, window_size=20, step_size=5):
    sequences, labels = [], []
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty or 'Label' not in df.columns:
                print(f"Skipping {csv_file}: File is empty or missing 'Label'.")
                continue

            feature_columns = [col for col in df.columns if col != 'Label']
            full_sequence = df[feature_columns].values.astype(np.float32)

            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]
                if window.shape != (window_size, len(feature_columns)):
                    continue

                window_labels = df['Label'][start:start + window_size]
                majority_label = Counter(window_labels).most_common(1)[0][0]

                sequences.append(window)
                labels.append(majority_label)

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)

# === Load Test Data ===
X_test, y_test = load_data_from_folder(test_data_folder)
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

if len(X_test) == 0:
    print("No valid sequences were found. Please check the data directory and file contents.")
    exit()

# Convert test data to PyTorch tensors and create DataLoader
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Evaluate the Model ===
all_preds, all_labels = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Calculate Test Accuracy ===
correct = sum(np.array(all_preds) == np.array(all_labels))
test_accuracy = correct / len(all_labels)
print(f"Test Accuracy ({model_type.upper()}): {test_accuracy * 100:.2f}%")

# === Confusion Matrix & Classification Report ===
conf_matrix = confusion_matrix(all_labels, all_preds)
behavior_labels = [behavior_mapping[i] for i in range(len(behavior_mapping))]

# Save Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=behavior_labels, yticklabels=behavior_labels)
plt.title(f"Confusion Matrix ({model_type.upper()})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
confusion_matrix_path = os.path.join(confusion_matrices_folder, f"confusion_matrix_{model_type}_{dataset_variant}.png")
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved as {confusion_matrix_path}")
plt.close()

# Save Classification Report
classification_report_text = classification_report(all_labels, all_preds, target_names=behavior_labels)
classification_report_path = os.path.join(classification_reports_folder, f"classification_report_{model_type}_{dataset_variant}.txt")
with open(classification_report_path, "w") as f:
    f.write(classification_report_text)
print(f"Classification report saved as {classification_report_path}")
