import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Network
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, num_layers):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.num_layers = num_layers

        # LSTM taking word embeddings as inputs and outputting hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)
        # optional: dropout

        self.linear = nn.Linear(self.hidden_dim*4, 64)
        # Linear layer maps from hidden state space to output space
        self.fc = nn.Linear(64, output_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output, _ = self.lstm(x)
        intermed = self.dropout(F.sigmoid(self.linear(output)))
        logits = self.fc(intermed)
        return logits