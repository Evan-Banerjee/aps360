# test.py

import torch

from utils.read_haikus import read_haikus_from_file
from utils.build_vocab import build_vocab
from model import HaikuLSTM
from generate import generate_haiku
import syllapy
import matplotlib.pyplot as plt


def load_vocab(haiku_file):
    """
    Reads haikus from the file and builds vocabulary mappings.

    Args:
        haiku_file (str): Path to the haikus text file.

    Returns:
        tuple: (word2idx, idx2word)
    """
    haikus = read_haikus_from_file(haiku_file)
    if not haikus:
        raise ValueError("No haikus found in the file. Please check the file format.")
    word2idx = build_vocab(haikus)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx, device):
    """
    Initializes the model and loads the saved state dictionary.

    Args:
        model_path (str): Path to the saved model checkpoint.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        hidden_dim (int): Dimension of LSTM hidden states.
        num_layers (int): Number of LSTM layers.
        pad_idx (int): Index of the padding token.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The loaded model.
    """
    model = HaikuLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx).to(device)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # If you have saved word2idx and idx2word in the checkpoint, load them here
    # Otherwise, ensure vocabulary is reconstructed correctly
    # word2idx = checkpoint.get('word2idx', None)
    # idx2word = checkpoint.get('idx2word', None)

    model.eval()  # Set the model to evaluation mode
    return model


def main():
    # Configuration
    haiku_file = 'test_data/haikus2-cleaned.txt'  # Path to your haikus text file
    model_path = 'models/haiku_model_final.pth'  # Path to the saved model checkpoint

    # Hyperparameters (must match those used during training)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Step 1: Load vocabulary
    print("Loading vocabulary...")
    word2idx, idx2word = load_vocab(haiku_file)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    # Step 2: Initialize and load the model
    print("Loading the model...")
    pad_idx = word2idx.get('<pad>', 0)  # Ensure '<pad>' exists
    model = load_model(
        model_path=model_path,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx,
        device=device
    )
    print("Model loaded successfully.")

    test_file = "test_data/output_test.txt"
    output_file = "test_data/sample_haikus_output.txt"


    syllables = []
    with open(test_file, 'r') as file:
        with open(output_file, 'w') as output:
            for line in file:
                line = line.strip()
                prompt = line
                generated_haiku = generate_haiku(model, word2idx, idx2word, prompt)
                output.write(generated_haiku + "\n\n")
                syllables.append(syllapy.count(generated_haiku))

    print(syllables)
    plt.figure(figsize=(10, 6))
    plt.hist(syllables, bins=range(min(syllables), max(syllables) + 1), edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.title("Histogram of Data Distribution")
    plt.xlabel("Element Value")
    plt.ylabel("Frequency")

    # Display the plot
    plt.grid(True)
    plt.savefig("plot.png", format="png", dpi=300)
    plt.show()



    # # Step 3: Prompt for haiku generation
    # prompt = input("Enter a prompt for haiku generation: ").strip()
    # if not prompt:
    #     prompt = "Whispering winds"  # Default prompt
    #     print(f"No prompt entered. Using default prompt: '{prompt}'")
    # while prompt != 'exit':
    #     # Step 4: Generate the haiku
    #     print("\nGenerating haiku...\n")
    #     generated_haiku = generate_haiku(model, word2idx, idx2word, prompt)
    #     print("Generated Haiku:")
    #     print(generated_haiku)
    #     prompt = input("Enter a prompt for haiku generation: ").strip()


if __name__ == "__main__":
    main()