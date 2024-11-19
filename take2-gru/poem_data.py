from torch.utils.data import Dataset

# inherit from torch
class PoemDataset(Dataset):

    def __init__(self, poems, word2idx):
        # need to convert the words to a list of indices
        self.data = []
        unknown_idx = word2idx.get('<unk>', None)
        for poem in poems:
            temp_poem = []
            words = poem.split()
            for word in words:
                encoding = word2idx.get(word, unknown_idx)
                temp_poem.append(encoding)
            self.data.append(temp_poem)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]