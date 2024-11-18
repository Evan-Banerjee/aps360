import torch.nn as nn

class HaikuGRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx, hidden_dim, num_layers, dropout, bidirectional):
        super(HaikuGRU, self).__init__()
        self.name = 'HaikuGRU'

        self.hidden_dim = hidden_dim
        self.output_dim = vocab_size
        self.number_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hidden_state):

        batch_size = x.size(0)

        x = self.embedding(x)

        x, hidden_state = self.gru(x, hidden_state)

        x = x.contiguous().view(-1, self.hidden_dim)

        x = self.fc(x)

        x = x.view(batch_size, -1, self.output_dim)

        return x, hidden_state

    def init_hidden(self, batch_size, device):
        weights = next(self.parameters()).data

        hidden_state = (weights.new(self.number_layers, batch_size, self.hidden_dim).zero_().to(device),
                        weights.new(self.number_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden_state