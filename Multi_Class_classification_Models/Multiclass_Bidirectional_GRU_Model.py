import torch
import torch.nn as nn


# Define Bi-GRU Model
class BiGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(BiGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
        h_n = h_n[-1]
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h_n)