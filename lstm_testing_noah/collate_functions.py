# collate_functions.py

from torch.nn.utils.rnn import pad_sequence


class CollateFn:
    """
    Collate function for DataLoader to handle padding of variable-length sequences.
    """

    def __init__(self, word2idx):
        """
        Initializes the collate function with a word-to-index mapping.

        Args:
            word2idx (dict): Mapping from word to index.
        """
        self.word2idx = word2idx

    def __call__(self, batch):
        """
        Pads input and target sequences in the batch.

        Args:
            batch (list of torch.Tensor): List of sequences (each sequence is a torch.Tensor).

        Returns:
            tuple: (input_padded, target_padded)
        """
        # Sort batch by sequence length in descending order (optional but recommended for RNNs)
        batch = sorted(batch, key=lambda x: len(x), reverse=True)

        input_sequences = [seq[:-1] for seq in batch]  # All tokens except last
        target_sequences = [seq[1:] for seq in batch]  # All tokens except first

        # Pad sequences
        input_padded = pad_sequence(input_sequences, batch_first=True, padding_value=self.word2idx['<pad>'])
        target_padded = pad_sequence(target_sequences, batch_first=True, padding_value=self.word2idx['<pad>'])

        return input_padded, target_padded