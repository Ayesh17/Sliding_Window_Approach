import torch
import torch.nn as nn


# Define RNN Model
class RNNClassifier(nn.Module):  # Renamed to reflect unidirectional nature
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)  # Unidirectional
        self.fc = nn.Linear(hidden_size, num_classes)  # Adjusted for unidirectional

    def forward(self, x):
        _, h_n = self.rnn(x)  # h_n shape: (num_layers, batch_size, hidden_size)
        h_n = h_n[-1]  # Take the last layer's hidden state (batch_size, hidden_size)
        return self.fc(h_n)
