# custom collate_fn

from torch.nn.utils.rnn import pad_sequence

def collate_fn(pad_idx, batch):
    # used to pad sequences to equal length

    # sort in descend order of size, helpful for our purposes
    batch = sorted(batch, key=lambda x: len(x), reverse=True)

    """
    Used to help the model predict the next word in a sequence.
    For example, if our poem is "the cat sat"
    input sequence will be "<start> the cat sat"
    and the target sequence will be "the cat sat <end>".
    """
    input_sequences = [seq[:-1] for seq in batch]  # All tokens except last
    target_sequences = [seq[1:] for seq in batch]  # All tokens except first

    # Pad the sequences with the index of the padding token
    input_padded = pad_sequence(input_sequences, batch_first=True, padding_value=pad_idx)
    target_padded = pad_sequence(target_sequences, batch_first=True, padding_value=pad_idx)

    return input_padded, target_padded
