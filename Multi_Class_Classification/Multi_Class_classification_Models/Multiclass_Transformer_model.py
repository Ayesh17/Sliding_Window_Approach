import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * (2 * torch.arange(0, d_model, 2).float() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    A Transformer-based model for multiclass sequence classification.
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

        # Learnable embedding layer that projects input features to d_model
        # shape: (batch_size, seq_len, input_size) -> (batch_size, seq_len, d_model)
        self.embedding = nn.Linear(input_size, d_model)

        # Positional encoding to add positional info
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Using batch_first = True for convenience
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        # Project features to d_model
        x = self.embedding(x)  # shape: (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # shape: (batch_size, seq_len, d_model)

        # Transformer Encoder
        # shape stays (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)

        # Use only the CLS-like token or pool over time for classification.
        # Strategy 1 (CLS-like): take the first time-step output
        # Strategy 2 (mean pool): average over sequence dimension
        # Here, let's do mean pooling:
        x = x.mean(dim=1)  # shape: (batch_size, d_model)

        # Classification
        logits = self.classifier(x)  # shape: (batch_size, num_classes)
        return logits
