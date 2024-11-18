import torch
import torch.nn as nn

from model import HaikuGRU

# There is no criteria to compare the models outputs against, so all data is for training.
def train(model, data_loader, criterion, optimizer, num_epochs, learning_rate, vocab_size, grad_norm, clip_grad, save_location, save_frequency, device):

    print(f'Using device: {device} for training')

    model.train()
    model.to(device)

    print('Starting Training')

    for epoch in range(num_epochs):
        total_loss = 0

        # init hidden state
        # the batch size might have to be the vocab size, not sure yet
        hidden_state = model.init_hidden(batch_size=data_loader.batch_size, device=device)

        print('Starting Batches')

        for (input, output) in data_loader:

            model.zero_grad(set_to_none=True)

            input.to(device)
            output.to(device)

            out, hidden_state = model(input, hidden_state)

            loss = criterion(out, output)

            loss.backward()

            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

            optimizer.step()

            total_loss += loss.item()

            print('Finished a batch')

        # print epoch info, and save the model
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {total_loss/len(data_loader)}')

        if (epoch+1) % save_frequency == 0:
            torch.save(model.state_dict(), save_location)
            print(f'Model saved to {save_location}')

    print('Training Complete')