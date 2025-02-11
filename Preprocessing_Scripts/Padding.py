import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import importlib


##############################################
# A) Preprocessing Function
##############################################
def filter_and_pad_csvs(
        input_folder,
        output_folder,
        min_frames=500,
        max_frames=1000
):
    """
    Processes each CSV in `input_folder` according to the rules:
      1. Skip files with fewer than `min_frames` rows.
      2. Truncate files with more than `max_frames` rows down to `max_frames`.
      3. If a file has between `min_frames` and `max_frames` rows, pad it up to `max_frames`,
         and set Label column of padded rows to -1 (so they can be ignored later).
    Saves processed files to `output_folder`.
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Gather all CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for csv_name in csv_files:
        input_path = os.path.join(input_folder, csv_name)

        try:
            df = pd.read_csv(input_path)

            nrows = len(df)
            if nrows < min_frames:
                # Skip files with fewer than 500 rows
                print(f"Skipping {csv_name}: Only {nrows} rows (< {min_frames}).")
                continue
            elif nrows > max_frames:
                # Truncate files to first 1000 rows
                df = df.iloc[:max_frames].copy()
                print(f"Truncating {csv_name} from {nrows} to {max_frames} rows.")
            else:
                # Between 500 and 1000 rows -> pad to 1000
                if nrows < max_frames:
                    pad_len = max_frames - nrows
                    # Create a zero DataFrame with same columns
                    zero_data = np.zeros((pad_len, df.shape[1]))
                    pad_df = pd.DataFrame(zero_data, columns=df.columns)
                    # Set label for padded rows to -1
                    if 'Label' in pad_df.columns:
                        pad_df['Label'] = -1

                    df = pd.concat([df, pad_df], ignore_index=True)
                    print(f"Padding {csv_name} from {nrows} to {max_frames} rows (Label for padded rows = -1).")

            # Save the processed file
            output_path = os.path.join(output_folder, csv_name)
            df.to_csv(output_path, index=False)
            print(f"Saved processed file: {csv_name}")

        except Exception as e:
            print(f"Error processing {csv_name}: {e}")


##############################################
# B) Loading Function that Skips -1 Windows
##############################################
def load_data_from_folder(
        folder_path,
        window_size=10,
        step_size=5
):
    """
    Loads CSV files from `folder_path`, then creates sliding windows of size `window_size`
    (stepped by `step_size`). If any row in a window has Label == -1, we skip that entire window.

    Returns:
      sequences: (num_sequences, window_size, num_features)
      labels: (num_sequences,)
    """
    sequences = []
    labels = []
    csv_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".csv")
    ]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Skip if empty or no Label column
            if df.empty or 'Label' not in df.columns:
                continue

            feature_columns = [col for col in df.columns if col != 'Label']
            full_sequence = df[feature_columns].values  # shape: (nrows, num_features)
            label_array = df['Label'].values  # shape: (nrows,)

            # Sliding-window approach
            for start in range(0, len(full_sequence) - window_size + 1, step_size):
                window_feats = full_sequence[start:start + window_size]
                window_labels = label_array[start:start + window_size]

                # Skip this window if any row has label == -1
                if (window_labels == -1).any():
                    continue

                # Majority label
                majority_label = Counter(window_labels).most_common(1)[0][0]

                sequences.append(window_feats)
                labels.append(majority_label)
        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)


##############################################
# C) Example Model(s)
#    (use your own LSTM/Transformer code)
##############################################
class SimpleExampleModel(nn.Module):
    """
    A trivial model just for demonstration; replace with your actual LSTM/Transformer, etc.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleExampleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, window_size, input_size)
        We'll just average over time dimension to get a single vector.
        """
        # average over time
        x = x.mean(dim=1)  # shape: (batch_size, input_size)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


##############################################
# D) Main Routine
##############################################
if __name__ == "__main__":
    # 1) Preprocess the data (train, validation, test)
    #    Adjust these paths to your actual folder structure
    dataset_variant = "Data_new"
    base_path = "../Datasets"

    # Input folders
    train_in = os.path.join(base_path, dataset_variant, "train")
    val_in = os.path.join(base_path, dataset_variant, "validation")
    test_in = os.path.join(base_path, dataset_variant, "test")  # optional

    # Output folders (filtered/padded)
    train_out = os.path.join(base_path, dataset_variant, "train_filtered")
    val_out = os.path.join(base_path, dataset_variant, "validation_filtered")
    test_out = os.path.join(base_path, dataset_variant, "test_filtered")

    # Apply filter and padding
    filter_and_pad_csvs(train_in, train_out, min_frames=500, max_frames=1000)
    filter_and_pad_csvs(val_in, val_out, min_frames=500, max_frames=1000)
    if os.path.exists(test_in):
        filter_and_pad_csvs(test_in, test_out, min_frames=500, max_frames=1000)

    # 2) Load the processed data with sliding windows
    window_size = 10
    step_size = 5

    X_train, y_train = load_data_from_folder(train_out, window_size, step_size)
    X_val, y_val = load_data_from_folder(val_out, window_size, step_size)

    print(f"Train sequences shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Val   sequences shape: {X_val.shape},   Val   labels shape: {y_val.shape}")

    # 3) Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 4) Instantiate the model (Example: a simple MLP)
    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    hidden_size = 128
    model = SimpleExampleModel(input_size, hidden_size, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5) Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # 6) Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, dim=1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, "
              f"Train Acc {train_accuracy:.4f}, Val Acc {val_accuracy:.4f}")

    # 7) (Optional) Evaluate on Test Set
    if os.path.exists(test_out):
        X_test, y_test = load_data_from_folder(test_out, window_size, step_size)
        if len(X_test) == 0:
            print("No valid test sequences found (maybe all were skipped?).")
        else:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, dim=1)
                    test_correct += (predicted == y_batch).sum().item()
                    test_total += y_batch.size(0)
            test_accuracy = test_correct / test_total
            print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        print("No test folder found. Skipping test evaluation.")
