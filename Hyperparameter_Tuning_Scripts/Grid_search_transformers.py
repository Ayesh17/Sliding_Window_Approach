import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib
from itertools import product

# Define the dataset variant
dataset_variant = "Data"
model_module = importlib.import_module("Multiclass_Transformer_model")
ModelClass = model_module.TransformerClassifier

# Define directories for training and validation CSV files
train_data_folder = f"../Datasets/{dataset_variant}/train"
val_data_folder = f"../Datasets/{dataset_variant}/validation"

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

X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)
X_train_tensor, y_train_tensor = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_val_tensor, y_val_tensor = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))
num_heads = 4
num_encoder_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid search hyperparameters
learning_rates = [0.0001, 0.0005, 0.001]
batch_sizes = [16, 32, 64, 128, 256]
dropouts = [0, 0.1, 0.2, 0.3, 0.4]
num_epochs = 30
early_stopping_patience = 10
best_val_accuracy = 0.0
best_params = None
best_model_state = None

for lr, batch_size, dropout in product(learning_rates, batch_sizes, dropouts):
    print(f"Training with LR={lr}, Batch Size={batch_size}, Dropout={dropout}")
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    model = ModelClass(input_size, num_classes, num_heads, num_encoder_layers, dim_feedforward=128, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        correct_preds, running_loss = 0, 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
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
            best_params = (lr, batch_size, dropout)
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping...")
            break

if best_model_state:
    model.load_state_dict(best_model_state)
    model_path = f"../Models/best_transformer_model_{dataset_variant}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved best model with LR={best_params[0]}, Batch Size={best_params[1]}, Dropout={best_params[2]} to {model_path}")
