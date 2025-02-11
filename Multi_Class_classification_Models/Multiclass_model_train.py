import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib  # For dynamic imports

# Define the dataset variant
dataset_variant = "Data"
model_type = "lstm"  # Change to "lstm" or "transformer"

# Dynamically import the appropriate model
if model_type == "lstm":
    model_module = importlib.import_module("Multiclass_LSTM_model")
    ModelClass = model_module.LSTMClassifier
elif model_type == "transformer":
    model_module = importlib.import_module("Multiclass_Transformer_model")
    ModelClass = model_module.TransformerClassifier
else:
    raise ValueError("Invalid model type. Choose 'lstm' or 'transformer'.")

# Define directories for training and validation CSV files based on dataset variant
train_data_folder = f"../Datasets/{dataset_variant}/train"
val_data_folder = f"../Datasets/{dataset_variant}/validation"

# Load data function
def load_data_from_folder(folder_path, window_size=20, step_size=10):
    sequences = []
    labels = []
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Ensure the dataframe has necessary columns
            if df.empty or 'Label' not in df.columns:
                continue

            feature_columns = [col for col in df.columns if col != 'Label']

            # Convert all data to float, coercing errors
            df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

            # Drop rows with NaN values after conversion
            df = df.dropna()

            full_sequence = df[feature_columns].values.astype(np.float32)  # Ensure float32

            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]
                if window.shape != (window_size, len(feature_columns)):
                    continue

                window_labels = df['Label'].iloc[start:start + window_size].values
                majority_label = Counter(window_labels).most_common(1)[0][0]

                sequences.append(window)
                labels.append(majority_label)

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)  # Ensure proper dtype

# Load training and validation data
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Set hyperparameters
input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))
learning_rate = 0.0001
hidden_size = 128 #128
num_layers = 2
num_heads = 4
num_encoder_layers = 2
dropout = 0.2

# Instantiate the selected model
if model_type == "lstm":
    model = ModelClass(input_size, hidden_size, num_layers, num_classes)
elif model_type == "transformer":
    model = ModelClass(input_size, num_classes, num_heads, num_encoder_layers, dim_feedforward=hidden_size, dropout=dropout)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100
early_stopping_patience = 10
best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct_preds = 0.0, 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()

    train_accuracy = correct_preds / len(train_loader.dataset)
    val_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    print(f"Epoch {epoch + 1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping...")
        break

# Save the best model
if best_model_state:
    model.load_state_dict(best_model_state)
    model_path = f"../Models/{model_type}_model_{dataset_variant}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved best model to {model_path}")
