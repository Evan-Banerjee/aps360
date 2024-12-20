from torch.utils.data import Dataset
import torch

# inherit from torch
class PoemDataset(Dataset):

    def __init__(self, poems, word2idx):
        # need to convert the words to a list of indices
        self.data = []
        unknown_idx = word2idx['<unk>']
        for poem in poems:
            temp_poem = []
            words = poem.split()
            for word in words:
                encoding = word2idx.get(word, unknown_idx)
                temp_poem.append(encoding)
            if len(temp_poem) == 0: # remove after done debugging
                print("Error oh my god why")
            self.data.append(torch.tensor(temp_poem, dtype=torch.long))
            if len(self.data) == 0:
                print("Something went wrong")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]