# dataset.py

import torch
from torch.utils.data import Dataset


class HaikuDataset(Dataset):
    """
    Custom Dataset for haikus.
    """

    def __init__(self, haikus, word2idx):
        """
        Initializes the dataset with haikus and a word-to-index mapping.

        Args:
            haikus (list of str): List of haikus.
            word2idx (dict): Mapping from word to index.
        """
        self.data = []
        for haiku in haikus:
            tokens = haiku.lower().split()
            indices = [word2idx.get(word, word2idx['<unk>']) for word in tokens]
            self.data.append(torch.tensor(indices, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        haiku = self.data[idx]
        return haiku