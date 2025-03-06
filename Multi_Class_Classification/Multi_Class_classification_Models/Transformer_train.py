import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset


###############################################################################
# A) Data Loading with Sliding Window and Label Remapping
###############################################################################
def load_data_from_folder(
        folder_path,
        window_size=10,
        step_size=5
):
    """
    Loads CSV files from `folder_path`. For each CSV (already padded/truncated to 1000 rows):
      1. We form sliding windows of length `window_size` in steps of `step_size`.
      2. If ANY row in the window is -1, we skip that window entirely.
      3. We find the majority label among the 10 rows. If it is -1, skip the window too.
      4. Store the window's features + the majority label.

    Returns:
      X: np.ndarray of shape (num_sequences, window_size, num_features)
      y: np.ndarray of shape (num_sequences,)
         (the raw labels, not remapped yet)
    """
    sequences = []
    labels = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'Label' not in df.columns:
                continue

            # All columns except 'Label' are features
            feature_cols = [c for c in df.columns if c != 'Label']
            data_array = df[feature_cols].values  # shape: (1000, num_features)
            label_array = df['Label'].values  # shape: (1000,)

            # Slide over windows
            for start in range(0, len(data_array) - window_size + 1, step_size):
                window_feats = data_array[start: start + window_size]
                window_labels = label_array[start: start + window_size]

                # If ANY row in the window is -1, skip
                if (window_labels == -1).any():
                    continue

                # Find majority label
                majority_label = Counter(window_labels).most_common(1)[0][0]
                if majority_label == -1:
                    # If the majority is still -1 for some reason, skip
                    continue

                sequences.append(window_feats)
                labels.append(majority_label)

        except Exception as e:
            print(f"Skipping {csv_file} due to error: {e}")

    X = np.array(sequences, dtype=np.float32)  # (num_sequences, window_size, num_features)
    y = np.array(labels, dtype=np.int64)  # (num_sequences,)

    return X, y


def create_label_mapping(*label_arrays):
    """
    Given one or more label arrays (e.g., y_train, y_val),
    find the union of all unique labels and map them to
    [0..num_classes-1]. Returns label2id, id2label dictionaries.
    """
    unique_labels = set()
    for arr in label_arrays:
        unique_labels.update(arr.tolist())
    unique_labels = sorted(list(unique_labels))  # sort them

    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return label2id, id2label


###############################################################################
# B) Positional Encoding for Transformer
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * (2.0 * torch.arange(0, d_model, 2).float() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


###############################################################################
# C) Transformer Classifier
###############################################################################
class TransformerClassifier(nn.Module):
    """
    A simple Transformer-based sequence classifier:
      - Linear embedding: (input_size) -> d_model
      - Positional encoding
      - Transformer encoder stack
      - Mean-pool over time dimension
      - Classification head -> num_classes
    """

    def __init__(
            self,
            input_size,
            num_classes,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=100
    ):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x)  # -> (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # -> (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # -> (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # -> (batch_size, d_model)
        logits = self.classifier(x)  # -> (batch_size, num_classes)
        return logits


###############################################################################
# D) Main Training Script
###############################################################################
def train_transformer_model():
    # -------------------------------------------------------------------------
    # 1) Paths to your filtered dataset
    # -------------------------------------------------------------------------
    dataset_variant = "Data_new"  # adjust as needed
    base_path = "../../Datasets"
    train_folder = os.path.join(base_path, dataset_variant, "train_filtered")
    val_folder = os.path.join(base_path, dataset_variant, "validation_filtered")
    test_folder = os.path.join(base_path, dataset_variant, "test_filtered")  # optional

    # Sliding window settings
    window_size = 10
    step_size = 5

    # -------------------------------------------------------------------------
    # 2) Load the train and validation sets
    # -------------------------------------------------------------------------
    X_train_raw, y_train_raw = load_data_from_folder(train_folder, window_size, step_size)
    X_val_raw, y_val_raw = load_data_from_folder(val_folder, window_size, step_size)

    print(f"Train sequences: {X_train_raw.shape}, labels: {y_train_raw.shape}")
    print(f"Val   sequences: {X_val_raw.shape},   labels: {y_val_raw.shape}")

    # -------------------------------------------------------------------------
    # 3) Build a label mapping to ensure labels are 0..n_classes-1
    #    This prevents out-of-range label errors in CrossEntropyLoss
    # -------------------------------------------------------------------------
    label2id, id2label = create_label_mapping(y_train_raw, y_val_raw)

    # Remap train labels
    y_train_mapped = np.array([label2id[lbl] for lbl in y_train_raw], dtype=np.int64)
    y_val_mapped = np.array([label2id[lbl] for lbl in y_val_raw], dtype=np.int64)

    # Possibly unify with test set too if you want to ensure all sets share the same mapping:
    # (We do this after we load test data, further below.)

    # -------------------------------------------------------------------------
    # 4) Create PyTorch Datasets
    # -------------------------------------------------------------------------
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_mapped, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val_raw, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_mapped, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # -------------------------------------------------------------------------
    # 5) Instantiate the Transformer
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train_raw.shape[2]
    # number of classes is the size of label2id
    num_classes = len(label2id)

    # Hyperparameters
    d_model = 64
    nhead = 4
    num_encoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    max_seq_len = window_size  # sequence length = 10

    model = TransformerClassifier(
        input_size=input_size,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------------------------------------------------
    # 6) Training Loop
    # -------------------------------------------------------------------------
    num_epochs = 20
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)  # (batch_size, num_classes)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = total_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_outs = model(X_batch)
                _, val_preds = torch.max(val_outs, dim=1)
                val_correct += (val_preds == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch [{epoch + 1}/{num_epochs}]  "
              f"Train Loss: {train_loss:.4f},  Train Acc: {train_acc:.4f},  "
              f"Val Acc: {val_acc:.4f}")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"Best Val Accuracy: {best_val_acc:.4f}")

    # -------------------------------------------------------------------------
    # 7) (Optional) Test Evaluation
    # -------------------------------------------------------------------------
    if os.path.exists(test_folder):
        X_test_raw, y_test_raw = load_data_from_folder(test_folder, window_size, step_size)
        if len(X_test_raw) == 0:
            print("No valid windows in the test set!")
            return model

        # If the test set might have labels not in train/val, unify them:
        # (Here we assume test labels are a subset, but to be safe do:)
        all_labels = set(y_train_raw.tolist()) | set(y_val_raw.tolist()) | set(y_test_raw.tolist())
        all_labels = sorted(list(all_labels))
        # Rebuild label2id if needed
        label2id = {lbl: idx for idx, lbl in enumerate(all_labels)}

        # Map test labels
        y_test_mapped = np.array([label2id[lbl] for lbl in y_test_raw], dtype=np.int64)

        X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_mapped, dtype=torch.long)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outs = model(X_batch)
                _, preds = torch.max(outs, dim=1)
                test_correct += (preds == y_batch).sum().item()
                test_total += X_batch.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0
        print(f"Test Accuracy: {test_acc:.4f}")
    else:
        print("No test folder found; skipping test evaluation.")

    return model


###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    trained_model = train_transformer_model()
