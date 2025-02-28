import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

from Multi_Class_classification_Models.Multiclass_Transformer_model import TransformerClassifier


# Load data function
def load_data_from_folder(folder_path, window_size=20, step_size=10):
    sequences = []
    labels = []
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

            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]
                if window.shape != (window_size, len(feature_columns)):
                    continue
                window_labels = df['Label'].iloc[start:start + window_size].values
                majority_label = np.bincount(window_labels).argmax()
                sequences.append(window)
                labels.append(majority_label)

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


# Define dataset paths
train_data_folder = "../Datasets/Data/train"
val_data_folder = "../Datasets/Data/validation"

# Load dataset
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# Define the parameter grid
param_grid = {
    "learning_rate": [0.001],
    "batch_size": [64],
    "dropout": [0.1, 0.2],
    "d_model": [128],
    "dim_feedforward": [128]
}

# Generate all possible combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Define input size and number of classes
input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))
num_heads = 4
num_encoder_layers = 2

# Track the best model and accuracy
best_val_accuracy = 0.0
best_params = None

# Perform grid search
for params in param_combinations:
    learning_rate, batch_size, dropout, d_model, dim_feedforward = params

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = TransformerClassifier(
        input_size=input_size,
        num_classes=num_classes,
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop (small number of epochs for fast evaluation)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        correct_preds = 0
        total_samples = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        train_accuracy = correct_preds / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Acc: {train_accuracy:.4f}")

    # Evaluate on validation set
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / len(val_dataset)
    print(f"Params: {params} -> Val Acc: {val_accuracy:.4f}")

    # Track best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params

print(f"Best Params: {best_params} -> Best Val Acc: {best_val_accuracy:.4f}")
