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


# Select Model Type & Dataset Variant
dataset_variant = "Data_1000"  # Change to "Data_1000" for padded dataset
model_type = "transformer"  # Options: rnn, bi_rnn, gru, bi_gru, lstm, bi_lstm, transformer

# Define Directory Paths
test_data_folder = f"../Datasets/{dataset_variant}/test"
confusion_matrices_folder = f"../Results/Confusion_Matrices"
classification_reports_folder = f"../Results/Classification_Reports"
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


# Detect Number of Features
def get_feature_count(folder_path):
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the test data folder.")

    df = pd.read_csv(csv_files[0])
    feature_columns = [col for col in df.columns if col != 'Label']
    return len(feature_columns)

no_of_features = get_feature_count(test_data_folder)
print(f"Detected number of features: {no_of_features}")


# Dynamically Import the Correct Model
model_mapping = {
    "rnn": "Multiclass_RNN_model.RNNClassifier",
    "bi_rnn": "Multiclass_Bidirectional_RNN_model.BiRNNClassifier",
    "gru": "Multiclass_GRU_model.GRUClassifier",
    "bi_gru": "Multiclass_Bidirectional_GRU_Model.BiGRUClassifier",
    "lstm": "Multiclass_LSTM_model.LSTMClassifier",
    "bi_lstm": "Multiclass_Bidirectional_LSTM_model.BiLSTMClassifier",
    "transformer": "Multiclass_Transformer_model.TransformerClassifier",
}

if model_type not in model_mapping:
    raise ValueError(f"Invalid model type '{model_type}'. Choose from {list(model_mapping.keys())}")

module_name, class_name = model_mapping[model_type].rsplit(".", 1)
model_module = importlib.import_module(f"Multi_Class_classification_Models.{module_name}")
ModelClass = getattr(model_module, class_name)


# Define Model Path & Load Model
model_path = f"../Models/{model_type}_model_{dataset_variant}.pth"
print("Model path:", model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate Model (Matching Training Parameters)
hidden_size = 128
num_layers = 2
dropout = 0.2

if "rnn" in model_type or "gru" in model_type or "lstm" in model_type:
    model = ModelClass(input_size=no_of_features, hidden_size=hidden_size, num_layers=num_layers, num_classes=len(behavior_mapping), dropout=dropout)
elif model_type == "transformer":
    d_model, num_heads, num_encoder_layers, dim_feedforward = 4, 2, 2, 128
    model = ModelClass(input_size=no_of_features, num_classes=len(behavior_mapping), d_model=d_model,
                       nhead=num_heads, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

# Load Trained Model Weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print(f"{model_type.upper()} Model loaded successfully.")


# Load Test Data with Sliding Windows
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

                window_labels = df['Label'][start:start + window_size].values

                # ✅ Skip sequences containing -1 labels (padded sequences)
                if (window_labels == -1).any():
                    continue

                majority_label = Counter(window_labels).most_common(1)[0][0]
                sequences.append(window)
                labels.append(majority_label)

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)

# Load Test Data
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


# Evaluate Model on Test Data
all_preds, all_labels = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Test Accuracy
correct = sum(np.array(all_preds) == np.array(all_labels))
test_accuracy = correct / len(all_labels)
print(f"Test Accuracy ({model_type.upper()}): {test_accuracy * 100:.2f}%")


# Confusion Matrix & Classification Report
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
