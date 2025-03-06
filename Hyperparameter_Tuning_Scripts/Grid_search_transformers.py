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


# Import models dynamically based on unidirectional or bidirectional choice
from Multi_Class_classification_Models.Multiclass_Transformer_model import TransformerClassifier

# Define dataset variant
dataset_variant = "Data_hyperparam"

# Define directories for training and validation CSV files
train_data_folder = f"../Datasets/{dataset_variant}/train"
val_data_folder = f"../Datasets/{dataset_variant}/validation"

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

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Load datasets
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Model parameters
input_size = X_train.shape[2]
num_classes = len(torch.unique(y_train))
num_heads = 4
num_encoder_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid search hyperparameters
# learning_rates = [0.0001, 0.0005, 0.001]
# batch_sizes = [16, 32, 64, 128, 256]
# dropouts = [0, 0.1, 0.2, 0.3, 0.4]

learning_rates = [0.0001, 0.001]
batch_sizes = [16, 32, 64]
dropouts = [0, 0.2, 0.4]

num_epochs = 30
early_stopping_patience = 10

# Initialize tracking
best_val_accuracy = 0.0
best_params = None
best_model_state = None
results = []

# Unique filename for results
def get_unique_filename(base_filename):
    filename = base_filename
    count = 1
    while os.path.exists(filename):
        filename = f"{base_filename.rsplit('.', 1)[0]}_{count}.csv"
        count += 1
    return filename

results_filename = get_unique_filename(f"../hyperparameter_results/Transformer_Results_{dataset_variant}.csv")

# Hyperparameter tuning loop
for lr, batch_size, dropout in product(learning_rates, batch_sizes, dropouts):
    print(f"\n🔹 Training with LR={lr}, Batch={batch_size}, Dropout={dropout}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    ModelClass = TransformerClassifier
    model = ModelClass(input_size, num_classes, num_heads, num_encoder_layers, dim_feedforward=128, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_train_accuracy = 0.0
    best_val_accuracy_run = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        correct_preds, total_loss, total = 0, 0.0, 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct_preds / total
        best_train_accuracy = max(best_train_accuracy, train_accuracy)

        # Validation step
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        best_val_accuracy_run = max(best_val_accuracy_run, val_accuracy)

        print(f" Epoch {epoch+1}/{num_epochs} - Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = (lr, batch_size, dropout)
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("🛑 Early stopping triggered.")
            break

    # Store results
    results.append({
        "learning_rate": lr,
        "batch_size": batch_size,
        "dropout": dropout,
        "best_training_accuracy": best_train_accuracy,
        "best_validation_accuracy": best_val_accuracy_run
    })

    # Save best model
    if best_model_state:
        model_save_path = f"../saved_models/Transformer_Best_{lr}_{batch_size}_{dropout}.pt"
        os.makedirs("../saved_models", exist_ok=True)
        torch.save(best_model_state, model_save_path)
        print(f"✅ Model saved: {model_save_path}")

# Save results to CSV
os.makedirs("../hyperparameter_results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(results_filename, index=False)

print(f"\n📂 Hyperparameter tuning results saved to: {results_filename}")
print(f"🎯 Best Hyperparameters: LR={best_params[0]}, Batch={best_params[1]}, Dropout={best_params[2]}")
