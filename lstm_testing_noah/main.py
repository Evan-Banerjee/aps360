# main.py

import os
from base64 import decode

import torch
from torch.utils.data import DataLoader
from utils.read_haikus import read_haikus_from_file
from utils.build_vocab import build_vocab
from dataset import HaikuDataset
from collate_functions import CollateFn
from model import HaikuLSTM
from train import train
from generate import generate_haiku


def main():
    # Paths
    haiku_file = 'test_data/haikus2-cleaned.txt'  # Ensure this file exists in your working directory
    final_model_path = 'models/haiku_model_final.pth'  # Path to save the final model
    checkpoint_dir = 'model_checkpoints'  # Directory to save checkpoints after each epoch

    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64  # Adjust based on your GPU memory
    num_workers = 4  # Adjust based on your CPU cores

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Step 1: Read haikus from file
    print("Reading haikus from file...")
    haikus = read_haikus_from_file(haiku_file)
    if not haikus:
        raise ValueError("No haikus found in the file. Please check the file format.")
    print(f"Loaded {len(haikus)} haikus.")

    # Step 2: Build vocabulary
    print("Building vocabulary...")
    word2idx = build_vocab(haikus)
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    # Step 3: Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = HaikuDataset(haikus, word2idx)
    collate_fn = CollateFn(word2idx)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Step 4: Initialize the model, criterion, and optimizer
    print("Initializing the model, criterion, and optimizer...")
    pad_idx = word2idx['<pad>']
    model = HaikuLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Train the model
    print("Starting training...")
    train(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=final_model_path,
        save_each_epoch=False,
        save_dir=checkpoint_dir
    )
    print("\nTraining completed. Generating a haiku...\n")
    # Step 6: Generate a haiku
    prompt = 'wind blows'
    generated_haiku = generate_haiku(model, word2idx, idx2word, prompt)
    print("Generated Haiku:")
    print(generated_haiku)


if __name__ == "__main__":
    main()