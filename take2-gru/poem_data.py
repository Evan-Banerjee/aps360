from torch.utils.data import Dataset

# inherit from torch
class PoemDataset(Dataset):

    def __init__(self, poems, word2idx):
        self.data = []
        #for poem in poems:
