import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Select Model Type & Dataset Variant
model_type = "rnn"  # Options: rnn, bi_rnn, gru, bi_gru, lstm, bi_lstm, transformer
dataset_variant = "Data_1000"  # Change to "Data_1000" for padded dataset



#########################################################################
#Also change the Model number because of the use of different hyperparams
#########################################################################



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

# Define Model Path & Load Model
model_path = f"../Models/{model_type}_model_{dataset_variant}_2.pth"
print("Model path:", model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Trained Model
torch.cuda.empty_cache()
model = torch.load(model_path, map_location=device, weights_only=False)
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

correct = sum(np.array(all_preds) == np.array(all_labels))
test_accuracy = correct / len(all_labels)
print(f"Test Accuracy ({model_type.upper()}): {test_accuracy * 100:.2f}%")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
behavior_labels = [behavior_mapping[i] for i in range(len(behavior_mapping))]

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=behavior_labels, yticklabels=behavior_labels)
plt.title(f"Confusion Matrix ({model_type.upper()})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
confusion_matrix_path = os.path.join(confusion_matrices_folder, f"confusion_matrix_{model_type}_{dataset_variant}.png")
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved as {confusion_matrix_path}")
plt.close()

# Generate Classification Report
classification_report_text = classification_report(all_labels, all_preds, target_names=behavior_labels)
classification_report_path = os.path.join(classification_reports_folder, f"classification_report_{model_type}_{dataset_variant}.txt")
with open(classification_report_path, "w") as f:
    f.write(classification_report_text)
print(f"Classification report saved as {classification_report_path}")
