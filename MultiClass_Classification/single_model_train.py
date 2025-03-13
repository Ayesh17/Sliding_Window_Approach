import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib

# 🔹 Ensure the script can find model modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ✅ Define dataset variant and model type
dataset_variant = "Data_1000"  # Change as needed
model_type = "transformer"  # Options: "lstm" or "transformer"

# 🔹 Dynamically import the appropriate model
model_mapping = {
    "lstm": "MultiClass_classification_Models.Multiclass_LSTM_model.LSTMClassifier",
    "transformer": "MultiClass_classification_Models.Multiclass_Transformer_model.TransformerClassifier"
}

if model_type not in model_mapping:
    raise ValueError(f"Invalid model type '{model_type}'. Choose from {list(model_mapping.keys())}")

module_name, class_name = model_mapping[model_type].rsplit(".", 1)
model_module = importlib.import_module(module_name)
ModelClass = getattr(model_module, class_name)

# ✅ Define directories for training and validation CSV files
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_data_folder = os.path.join(base_dir, "Datasets", dataset_variant, "train")
val_data_folder = os.path.join(base_dir, "Datasets", dataset_variant, "validation")

# 🔹 Ensure dataset directories exist
if not os.path.exists(train_data_folder) or not os.path.exists(val_data_folder):
    raise FileNotFoundError(f"Dataset folders not found: {train_data_folder} or {val_data_folder}")

# ✅ Load data function with padding handling
def load_data_from_folder(folder_path, window_size=20, step_size=10):
    sequences, labels = [], []
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

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
                if window.shape != (window_size, len(feature_columns)):
                    continue

                window_labels = label_array[start:start + window_size]

                # 🔹 Ignore padding labels (-1) when selecting the majority class
                non_padding_labels = window_labels[window_labels >= 0]
                if len(non_padding_labels) == 0:
                    continue  # Skip if all labels in the window are padding

                majority_label = Counter(non_padding_labels).most_common(1)[0][0]
                sequences.append(window)
                labels.append(majority_label)

        except Exception as e:
            print(f"Skipping {csv_file}: {e}")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

# ✅ Load training and validation data
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)

# 🔹 Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ✅ Set model hyperparameters
input_size = X_train.shape[2]
num_classes = len(np.unique(y_train[y_train >= 0]))  # Exclude padding labels from class count
learning_rate = 0.001
# hidden_size = 128
num_layers = 2
num_heads = 4
num_encoder_layers = 2
dropout = 0

# ❗ Debugging: Print unique labels
print("Unique training labels:", np.unique(y_train))
print("Unique validation labels:", np.unique(y_val))

# 🔹 Instantiate the selected model
if model_type == "lstm":
    model = ModelClass(input_size, hidden_size, num_layers, num_classes, dropout=dropout)
elif model_type == "transformer":
    model = ModelClass(input_size, num_classes, num_heads, num_encoder_layers, dropout=dropout)

# ✅ Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 🔹 Define loss and optimizer (ignore padding labels)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training loop with early stopping
num_epochs = 10
early_stopping_patience = 10
best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct_preds, total_samples = 0.0, 0, 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)

        # 🔹 Ignore padding labels (-1) in accuracy calculation
        mask = labels >= 0
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted[mask] == labels[mask]).sum().item()
        total_samples += mask.sum().item()

    train_accuracy = correct_preds / total_samples if total_samples > 0 else 0

    # 🔹 Validation phase
    val_loss, val_correct, val_total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * sequences.size(0)

            mask = labels >= 0
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted[mask] == labels[mask]).sum().item()
            val_total += mask.sum().item()

    val_accuracy = val_correct / val_total if val_total > 0 else 0
    print(f"Epoch {epoch + 1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    # 🔹 Early stopping logic
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping...")
        break

# ✅ Save the best model
if best_model_state:
    model.load_state_dict(best_model_state)

    models_dir = os.path.join(base_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{model_type}_model_{dataset_variant}.pth")

    # 🔹 Save the entire model instead of just state_dict
    torch.save(model, model_path)

    print(f"Saved best model to {model_path}")
