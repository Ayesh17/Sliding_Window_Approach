import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.layer_norm = nn.LayerNorm(hidden_size)  # LayerNorm for stability
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)  # Adjusted for unidirectional

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)
        out = torch.mean(lstm_out, dim=1)  # Mean pooling
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
