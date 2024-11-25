import torch.nn as nn
import torch

class HaikuGRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx, hidden_dim, num_layers, dropout, bidirectional):
        super(HaikuGRU, self).__init__()
        self.name = 'HaikuGRU'

        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim * self.num_directions
        self.output_dim = vocab_size
        self.number_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=vocab_size)

    def forward(self, x, hidden_state):

        batch_size = x.size(0)

        x = self.embedding(x)

        x, hidden_state = self.gru(x, hidden_state)

        x = x.contiguous().view(-1, self.hidden_dim)

        x = self.fc(x)

        x = x.view(batch_size, -1, self.output_dim)

        return x, hidden_state

    def init_hidden(self, batch_size, device):
        #weights = next(self.parameters()).data

        #hidden_state = (weights.new(self.number_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions).zero_().to(device))

        hidden_state = torch.zeros(self.number_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions).to(device)

        return hidden_state