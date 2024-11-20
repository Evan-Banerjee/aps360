import torch
import torch.nn as nn
import os


# There is no criteria to compare the models outputs against, so all data is for training.
def train(model, data_loader, criterion, optimizer, num_epochs, learning_rate, vocab_size, grad_norm, clip_grad,
          save_location, save_frequency, device, model_params):
    """
    Trains the model
    :param model:
    :param data_loader:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :param learning_rate:
    :param vocab_size:
    :param grad_norm:
    :param clip_grad:
    :param save_location:
    :param save_frequency:
    :param device:
    :param model_params:
    :return: Trained Model
    """

    print(f'Using device: {device} for training')

    model.to(device)
    model.train()


    print('Starting Training')

    for epoch in range(num_epochs):
        total_loss = 0

        # init hidden state
        # the batch size might have to be the vocab size, not sure yet

        print('Starting Batches')

        for (in_seq, out_seq) in data_loader:

            model.zero_grad(set_to_none=True)

            in_seq = in_seq.to(device)
            out_seq = out_seq.to(device)

            hidden_state = model.init_hidden(batch_size=in_seq.size(0), device=device)

            out, hidden_state = model(in_seq, hidden_state)

            out = out.view(-1, out.size(2))  # (batch_size * seq_len, vocab_size)
            out_seq = out_seq.view(-1)

            loss = criterion(out, out_seq)

            loss.backward()

            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

            optimizer.step()

            total_loss += loss.item()

            print('Finished a batch')

        # print epoch info, and save the model
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Loss: {total_loss / len(data_loader)}')

        if (epoch + 1) % save_frequency == 0:
            save_file = os.path.join(save_location, f'haiku_model_epoch_{epoch + 1}.pth')
            config = {name: value for name, value in model_params}
            model_info = {
                'state_dict': model.state_dict(),
                'config': config
            }
            torch.save(model_info, save_file)
            print(f'Model saved to {save_location}')

    print('Training Complete')
    return model
