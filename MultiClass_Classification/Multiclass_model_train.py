import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import csv
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib
import re

# ✅ Parse Command-line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--dataset_variant", type=str, required=True)
# parser.add_argument("--hidden_size", type=int, required=True)
parser.add_argument("--dropout", type=float, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args = parser.parse_args()

# ✅ Set Hyperparameters from CLI Arguments
model_type = args.model_type
dataset_variant = args.dataset_variant
# hidden_size = args.hidden_size
dropout = args.dropout
learning_rate = args.learning_rate
batch_size = args.batch_size

# Get the absolute path of the project root directory (two levels up)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the correct dataset path
train_data_folder = os.path.join(base_dir, "Datasets", dataset_variant, "train")
val_data_folder = os.path.join(base_dir, "Datasets", dataset_variant, "validation")


# Function to Load Data
def load_data_from_folder(folder_path, window_size=20, step_size=10):
    sequences, labels = [], []
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty or 'Label' not in df.columns:
                continue

            feature_columns = [col for col in df.columns if col != 'Label']
            df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
            df = df.dropna()

            full_sequence = df[feature_columns].values.astype(np.float32)
            label_array = df['Label'].values.astype(np.int64)

            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]
                window_labels = label_array[start:start + window_size]

                if (window_labels == -1).any():
                    continue

                majority_label = Counter(window_labels).most_common(1)[0][0]
                sequences.append(window)
                labels.append(majority_label)
        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


# Load Data
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model Configuration
input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))

# Dynamically Import Model
model_mapping = {
    "rnn": "Multiclass_RNN_model.RNNClassifier",
    "bi_rnn": "Multiclass_Bidirectional_RNN_model.BiRNNClassifier",
    "gru": "Multiclass_GRU_Model.GRUClassifier",
    "bi_gru": "Multiclass_Bidirectional_GRU_Model.BiGRUClassifier",
    "lstm": "Multiclass_LSTM_model.LSTMClassifier",
    "bi_lstm": "Multiclass_Bidirectional_LSTM_model.BiLSTMClassifier",
    "transformer": "Multiclass_Transformer_model.TransformerClassifier",
}

if model_type not in model_mapping:
    raise ValueError(f"Invalid model type '{model_type}'. Choose from {list(model_mapping.keys())}")

module_name, class_name = model_mapping[model_type].rsplit(".", 1)
model_module = importlib.import_module(f"MultiClass_classification_Models.{module_name}")
ModelClass = getattr(model_module, class_name)

# Instantiate Model
if "rnn" in model_type or "gru" in model_type or "lstm" in model_type:
    model = ModelClass(input_size, hidden_size, num_layers=2, num_classes=num_classes, dropout=dropout)
elif "transformer" in model_type:
    model = ModelClass(input_size, num_classes, num_encoder_layers=2, dropout=dropout)

# Move Model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop with Early Stopping
num_epochs = 100
early_stopping_patience = 10
best_val_accuracy = 0.0
best_model = None
epochs_without_improvement = 0
best_epoch = 0
train_accuracy_at_best_epoch = 0.0

for epoch in range(num_epochs):
    model.train()
    correct_preds = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()

    train_accuracy = correct_preds / len(train_loader.dataset)

    # Validation phase
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    print(f"Epoch {epoch + 1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model
        best_epoch = epoch + 1
        train_accuracy_at_best_epoch = train_accuracy
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping...")
        break

# Ensure Models directory exists
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)


# Increment model filename
base_model_path = os.path.join(models_dir, f"{model_type}_model_{dataset_variant}")
existing_models = [f for f in os.listdir(models_dir) if f.startswith(f"{model_type}_model_{dataset_variant}")]
existing_versions = [int(re.search(r"_(\d+)\.pth$", f).group(1)) for f in existing_models if re.search(r"_(\d+)\.pth$", f)]
next_version = max(existing_versions) + 1 if existing_versions else 1
best_model_path = f"{base_model_path}_{next_version}.pth"

# Save the Best Model
torch.save(best_model, best_model_path)
print(f"Saved full best model to {best_model_path}")

# Save training log
model_filename = os.path.basename(best_model_path)  # Extract filename from full path
log_csv_path = os.path.join(models_dir, "training_log.csv")
log_data = {
    "model_type": model_type,
    "dataset_variant": dataset_variant,
    # "hidden_size": hidden_size,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "train_accuracy_at_best_epoch": round(train_accuracy_at_best_epoch, 4),
    "best_val_accuracy": round(best_val_accuracy, 4),
    "best_epoch": best_epoch,
    "model_name": model_filename  # Store the full filename
}
write_header = not os.path.exists(log_csv_path)

with open(log_csv_path, mode="a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=log_data.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(log_data)

print(f"Training log updated: {log_csv_path}")