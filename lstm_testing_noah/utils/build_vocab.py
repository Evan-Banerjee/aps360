# utils/build_vocab.py

from collections import Counter

def build_vocab(haikus):
    """
    Builds a vocabulary dictionary from the haikus.

    Args:
        haikus (list of str): List of haikus.

    Returns:
        dict: Mapping from word to index.
    """
    words = []
    for haiku in haikus:
        words.extend(haiku.lower().split())
    word_counts = Counter(words)
    vocab = ['<pad>', '<unk>', '<eos>'] + sorted(word_counts.keys())
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx