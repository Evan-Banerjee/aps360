# train.py

import os
import torch
from torch.cuda.amp import GradScaler, autocast


def train(model, data_loader, criterion, optimizer, num_epochs, save_path=None, save_each_epoch=False, save_dir=None,
          max_grad_norm=5.0):
    """
    Training loop for the model with mixed precision and gradient clipping.
    Optionally saves the model after each epoch or after all epochs.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs to train.
        save_path (str, optional): Path to save the final model.
        save_each_epoch (bool, optional): Whether to save the model after each epoch.
        save_dir (str, optional): Directory to save epoch-wise checkpoints.
        max_grad_norm (float, optional): Maximum norm for gradient clipping.
    """
    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()
    model.train()

    # Determine the device from the model's parameters
    device = next(model.parameters()).device
    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(data_loader):
            # Move data to the appropriate device
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()
            hidden = None  # Let LSTM initialize hidden state as zeros

            # Forward pass with autocasting for mixed precision
            with autocast():
                output, hidden = model(input_seq, hidden)
                # Reshape output and target for loss computation
                output = output.view(-1, output.size(2))  # (batch_size * seq_len, vocab_size)
                target_seq = target_seq.view(-1)  # (batch_size * seq_len)
                loss = criterion(output, target_seq)

            # Backward pass and optimization with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save model checkpoint after each epoch if enabled
        if save_each_epoch and save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'scaler_state_dict': scaler.state_dict()
            }
            epoch_save_path = os.path.join(save_dir, f'haiku_model_epoch_{epoch + 1}.pth')
            torch.save(checkpoint, epoch_save_path)
            print(f"Checkpoint saved to {epoch_save_path}")

    # Save the final model checkpoint if save_path is provided
    if save_path is not None:
        checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f"Final checkpoint saved to {save_path}")