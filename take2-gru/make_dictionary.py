from collections import Counter

def make_dictionary(poems):
    """
    Create a word2idx and idx2word dictionary of all words in all poems.
    Reserve the first three tokens for padding, unknown, and end of sequence
    :param poems:
    :return: word2idx, idx2word
    """

    word2idx = {}
    idx2word = {}

    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    word2idx['<eos>'] = 2
    idx2word[0] = '<pad>'
    idx2word[1] = '<unk>'
    idx2word[2] = '<eos>'

    count = 3

    min_frequency = 2

    word_frequencies = Counter()

    for poem in poems:
        words = poem.split()
        for word in words:
            word_frequencies[word.lower()] += 1

    for word, frequency in word_frequencies.items():
        if frequency >= min_frequency:
            if word not in word2idx:
                word2idx[word] = count
                idx2word[count] = word
                count += 1


    # for poem in poems:
    #     # Get the word out of the string
    #     words = poem.split()
    #     for word in words:
    #         word = word.lower()
    #         if word not in word2idx:
    #             word2idx[word] = count
    #             idx2word[count] = word
    #             count += 1



    return word2idx, idx2word

