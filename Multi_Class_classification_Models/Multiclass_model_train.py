import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib

# Define the dataset variant (options: Data, Data_1000)
dataset_variant = "Data_1000"

# Choose model type (options: rnn, gru, lstm, bi_rnn, bi_gru, bi_lstm, transforme)
model_type = "bi_lstm"

# Define the module and class names for different models
model_mapping = {
    "rnn": "Multiclass_RNN_model.RNNClassifier",
    "bi_rnn": "Multiclass_Bidirectional_RNN_model.BiRNNClassifier",
    "gru": "Multiclass_GRU_model.GRUClassifier",
    "bi_gru": "Multiclass_Bidirectional_GRU_Model.BiGRUClassifier",
    "lstm": "Multiclass_LSTM_model.LSTMClassifier",
    "bi_lstm": "Multiclass_Bidirectional_LSTM_model.BiLSTMClassifier",
    "transformer": "Multiclass_Transformer_model.TransformerClassifier",
}

# Validate model type
if model_type not in model_mapping:
    raise ValueError(f"Invalid model type '{model_type}'. Choose from {list(model_mapping.keys())}")

# Dynamically import the correct model
module_name, class_name = model_mapping[model_type].rsplit(".", 1)
model_module = importlib.import_module(f"Multi_Class_classification_Models.{module_name}")
ModelClass = getattr(model_module, class_name)

# Define directories for training and validation CSV files
train_data_folder = f"../Datasets/{dataset_variant}/train"
val_data_folder = f"../Datasets/{dataset_variant}/validation"


# Function to load data with a sliding window approach
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
hidden_size = 128
num_layers = 2
num_heads = 4
num_encoder_layers = 2
dropout = 0.2

# Instantiate the selected model
if "rnn" in model_type or "gru" in model_type or "lstm" in model_type:
    model = ModelClass(input_size, hidden_size, num_layers, num_classes, dropout=dropout)
elif "transformer" in model_type:
    model = ModelClass(input_size, num_classes, num_heads, num_encoder_layers, dim_feedforward=hidden_size,
                       dropout=dropout)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping
num_epochs = 2  # Increased for better learning
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

    # Validation phase
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

    # Check for early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping...")
        break

# Ensure Models directory exists
os.makedirs("../Models", exist_ok=True)

# Create unique filename if model already exists
base_model_path = f"../Models/{model_type}_model_{dataset_variant}.pth"
model_path = base_model_path
counter = 1
while os.path.exists(model_path):
    model_path = f"../Models/{model_type}_model_{dataset_variant}_{counter}.pth"
    counter += 1

# Save the best model
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), model_path)
print(f"Saved best model to {model_path}")
