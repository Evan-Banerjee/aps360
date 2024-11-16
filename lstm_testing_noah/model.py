# model.py

import torch.nn as nn


class HaikuLSTM(nn.Module):
    """
    LSTM model for haiku generation.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx):
        """
        Initializes the HaikuLSTM model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            pad_idx (int): Index of the padding token.
        """
        super(HaikuLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            hidden (tuple): Tuple of (h_0, c_0) for LSTM.

        Returns:
            tuple: (output logits, hidden state)
        """
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        out, hidden = self.lstm(emb, hidden)  # out: (batch_size, seq_len, hidden_dim)
        out = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return out, hidden



class HaikuGRU(nn.Module):
    """
    LSTM model for haiku generation.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx):
        """
        Initializes the HaikuLSTM model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            pad_idx (int): Index of the padding token.
        """
        super(HaikuGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            hidden (tuple): Tuple of (h_0, c_0) for LSTM.

        Returns:
            tuple: (output logits, hidden state)
        """
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        out, hidden = self.gru(emb, hidden)  # out: (batch_size, seq_len, hidden_dim)
        out = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return out, hidden