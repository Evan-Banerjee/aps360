from create_dataset import create_dataset
from make_dictionary import make_dictionary
from poem_data import PoemDataset
from model import HaikuGRU

import nltk
from nltk.corpus import cmudict
from torchinfo import summary
import torch

def main():
    poems_path = 'data/poems-cleaned-poems.txt'
    poems = create_dataset(poems_path)

    word2idx, idx2word = make_dictionary(poems)
    assert len(word2idx) == len(idx2word)
    print(f'Length of dictionary: {len(word2idx)} words')
    vocab_size = len(idx2word)
    padding_idx = word2idx.get('<pad>', None)
    unknown_idx = word2idx.get('<unk>', None)

    # Hyperparameters ----
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2
    dropout = 0
    bidirectional = False

    model = HaikuGRU(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)




if __name__ == '__main__':
    main()