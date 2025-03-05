import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from Multiclass_Bidirectional_LSTM_model import LSTMClassifier

# Define the dataset variant
dataset_variant = "Data15"  # Change this variable for each dataset variant

# Define directories for training and validation CSV files based on dataset variant
train_data_folder = f"../Datasets/{dataset_variant}/train"
val_data_folder = f"../Datasets/{dataset_variant}/validation"

# Ensure folders for saving accuracy and loss plots exist
accuracy_plots_folder = f"../Plots/Accuracy_plots"
os.makedirs(accuracy_plots_folder, exist_ok=True)

loss_plots_folder = f"../Plots/Loss_plots"
os.makedirs(loss_plots_folder, exist_ok=True)

# Ensure a folder for saving models exists
models_folder = "../Models"
os.makedirs(models_folder, exist_ok=True)


# Load data function
def load_data_from_folder(folder_path, window_size=20, step_size=5):
    sequences = []
    labels = []

    # Collect all CSV files in the folder
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Skip files that are empty or don't have the expected 'Label' column
            if df.empty or 'Label' not in df.columns:
                print(f"Skipping {csv_file}: File is empty or 'Label' column not found.")
                continue

            # Extract feature columns (all columns except 'Label')
            feature_columns = [col for col in df.columns if col != 'Label']
            full_sequence = df[feature_columns].values  # Convert features to a NumPy array

            # Generate sliding windows
            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window = full_sequence[start:start + window_size]

                # Skip if window size is not met
                if window.shape != (window_size, len(feature_columns)):
                    continue

                # Append the window and label
                sequences.append(window)
                labels.append(df['Label'].iloc[0])  # Assuming the label is constant

        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)


# Load training and validation data
X_train, y_train = load_data_from_folder(train_data_folder)
X_val, y_val = load_data_from_folder(val_data_folder)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Set hyperparameters
input_size = X_train.shape[2]  # Number of features
print("input_size", input_size)
hidden_size = 128
num_layers = 2
num_classes = len(np.unique(y_train))  # Number of unique classes
num_epochs = 100
learning_rate = 0.001
early_stopping_patience = 10

# Calculate dynamic class weights
unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weights = total_samples / (len(unique_classes) * class_counts)  # Inverse frequency weighting


# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust weight for "headon" class
headon_class_index = 1  # Replace with the actual index of the "headon" class
class_weights[headon_class_index] *= 0.5  # Reduce the weight for "headon" (tune this factor as needed)

# Convert to PyTorch tensor and move to device
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Initialize the model, loss function, and optimizer
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

# Training loop with early stopping and best model saving
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
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

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_accuracy = correct_preds / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Validation loop
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_accuracy = val_correct / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

    # Check if this is the best model so far
    if epoch_val_accuracy > best_val_accuracy:
        best_val_accuracy = epoch_val_accuracy
        best_model_state = model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Early stopping condition
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch + 1} due to no improvement for {early_stopping_patience} epochs.")
        break

# Save the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model_path = os.path.join(models_folder, f"lstm_model_{dataset_variant}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved as '{model_path}'")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title(f'Model Accuracy for {dataset_variant}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
accuracy_plot_path = os.path.join(accuracy_plots_folder, f"accuracy_plot_{dataset_variant}.png")
plt.savefig(accuracy_plot_path)
print(f"Accuracy plot saved as {accuracy_plot_path}")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title(f'Model Loss for {dataset_variant}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
loss_plot_path = os.path.join(loss_plots_folder, f"loss_plot_{dataset_variant}.png")
plt.savefig(loss_plot_path)
print(f"Loss plot saved as {loss_plot_path}")
plt.close()
