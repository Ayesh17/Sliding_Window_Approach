import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from Multiclass_LSTM_model import LSTMClassifier

# Define the directory for testing CSV files
test_data_folder = "../Data/test"

# Behavior label mapping
behavior_mapping = {
    0: "benign",
    1: "block",
    2: "ram",
    3: "cross",
    4: "headon",
    5: "herd",
    6: "overtake"
}

# Load trained model
model_path = "lstm_ship_behavior_model_multiclass.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_size=23, hidden_size=128, num_layers=2, num_classes=len(behavior_mapping))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

# Function to load test data with sliding windows
def load_data_from_folder(folder_path, window_size=20, step_size=5):
    sequences = []
    labels = []

    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if df.empty or 'Label' not in df.columns:
                print(f"Skipping {csv_file}: File is empty or 'Label' column not found.")
                continue

            feature_columns = [col for col in df.columns if col != 'Label']
            full_sequence = df[feature_columns].values

            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]

                if window.shape != (window_size, len(feature_columns)):
                    continue

                sequences.append(window)
                labels.append(df['Label'].iloc[0])

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)

# Load testing data
X_test, y_test = load_data_from_folder(test_data_folder)
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Check if valid sequences are loaded
if len(X_test) == 0:
    print("No valid sequences were found. Please check the data directory and file contents.")
    exit()

# Convert test data to PyTorch tensors and create DataLoader
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model on test data
all_preds, all_labels = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate test accuracy
correct = sum(np.array(all_preds) == np.array(all_labels))
test_accuracy = correct / len(all_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
behavior_labels = [behavior_mapping[i] for i in range(len(behavior_mapping))]

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=behavior_labels, yticklabels=behavior_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=behavior_labels))
