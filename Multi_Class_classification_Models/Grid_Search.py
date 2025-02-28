import itertools
import os
import importlib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Define hyperparameter search space
hyperparameter_grid = {
    "learning_rate": [0.001, 0.0001, 0.0005],
    "batch_size": [16, 32, 64, 128],
    "dropout": [0, 0.1, 0.2, 0.3, 0.4],
    "hidden_size": [32, 64, 128]
}

# Generate all hyperparameter combinations
hyperparameter_combinations = list(itertools.product(
    hyperparameter_grid["learning_rate"],
    hyperparameter_grid["batch_size"],
    hyperparameter_grid["dropout"],
    hyperparameter_grid["hidden_size"]
))


# Load data function
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
            df.dropna(inplace=True)

            full_sequence = df[feature_columns].values.astype(np.float32)
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

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


# Load training and validation data
def load_dataset(dataset_variant):
    train_folder = f"../Datasets/{dataset_variant}/train"
    val_folder = f"../Datasets/{dataset_variant}/validation"
    return load_data_from_folder(train_folder), load_data_from_folder(val_folder)


# Grid search function
def grid_search(dataset_variant="Data_1000"):
    # Preload dataset once to avoid redundant loading
    (X_train, y_train), (X_val, y_val) = load_dataset(dataset_variant)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_hyperparams, best_accuracy = None, 0.0
    model_module = importlib.import_module("Multiclass_LSTM_model")
    ModelClass = model_module.LSTMClassifier

    for lr, batch_size, dropout, hidden_size in hyperparameter_combinations:
        print(f"Testing: lr={lr}, batch_size={batch_size}, dropout={dropout}, hidden_size={hidden_size}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ModelClass(input_size, hidden_size, num_layers=2, num_classes=num_classes, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_accuracy, early_stopping_patience, epochs_without_improvement = 0.0, 10, 0

        for epoch in range(30):  # Training loop

            model.train()
            for batch_idx, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(sequences), labels)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()  # Free unused GPU memory
            print(f"Epoch {epoch + 1}/30 - Training Completed")  # Epoch progress update

            model.eval()
            val_correct = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
            val_accuracy = val_correct / len(val_loader.dataset)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy, epochs_without_improvement = val_accuracy, 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(
            f"Validation Accuracy: {best_val_accuracy:.4f}, Hyperparameters: lr={lr}, batch_size={batch_size}, dropout={dropout}, hidden_size={hidden_size}")
        if best_val_accuracy > best_accuracy:
            best_accuracy, best_hyperparams = best_val_accuracy, (lr, batch_size, dropout, hidden_size)

    if best_hyperparams:
        print(
            f"Best Hyperparameters: Learning Rate={best_hyperparams[0]}, Batch Size={best_hyperparams[1]}, Dropout={best_hyperparams[2]}, Hidden Size={best_hyperparams[3]}")

    else:
        print("No valid hyperparameter combination found.")


# Run grid search
grid_search()
