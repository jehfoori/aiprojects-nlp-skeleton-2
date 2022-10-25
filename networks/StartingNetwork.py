import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Network
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM taking word embeddings as inputs and outputting hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)
        # optional: dropout

        # Linear layer maps from hidden state space to output space
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        logits = self.fc(output)
        return F.sigmoid(logits), state