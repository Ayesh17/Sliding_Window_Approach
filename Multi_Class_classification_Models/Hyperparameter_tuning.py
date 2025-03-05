import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from Multiclass_Bidirectional_LSTM_model import LSTMClassifier

# Define directories for training and validation CSV files
train_data_folder = "../Data15/train"
val_data_folder = "../Data15/validation"


# Load data function
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
                if window.shape == (window_size, len(feature_columns)):
                    sequences.append(window)
                    labels.append(df['Label'].iloc[0])

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)


# Load data
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Hyperparameter grid
param_grid = {
    "hidden_size": [64, 128],
    "num_layers": [1, 2],
    "learning_rate": [0.001, 0.0001],
    "batch_size": [32, 64]
}

input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))
num_epochs = 10  # Reduced epochs for quicker tuning


# Function to train and evaluate the model
def train_and_evaluate(hidden_size, num_layers, learning_rate, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    return val_accuracy


# Grid search over hyperparameters
best_accuracy = 0
best_params = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for hidden_size in param_grid["hidden_size"]:
    for num_layers in param_grid["num_layers"]:
        for learning_rate in param_grid["learning_rate"]:
            for batch_size in param_grid["batch_size"]:
                val_accuracy = train_and_evaluate(hidden_size, num_layers, learning_rate, batch_size)
                print(f"Params: hidden_size={hidden_size}, num_layers={num_layers}, "
                      f"learning_rate={learning_rate}, batch_size={batch_size}, "
                      f"Validation Accuracy: {val_accuracy:.4f}")

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_params = {
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size
                    }

print(f"Best params: {best_params} with Validation Accuracy: {best_accuracy:.4f}")
