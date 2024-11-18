import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Define your HaikuGRU model as previously discussed
class HaikuGRU(nn.Module):
    # [Model definition as above]
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(HaikuGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        out = out.reshape(-1, out.shape[2])
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


# Dummy implementations for preprocessing and dataset
def preprocess_poems(data_path):
    poems = []
    with open(data_path, 'r') as f:
        i = 0
        for line in f:
            if line.startswith('\n') != 1:
                poems.append(line)

    return poems


def build_vocab(poems):
    # Simple word to index mapping
    word2idx = {}
    idx = 0
    for poem in poems:
        for word in poem.split():
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    return word2idx


class PoemDataset(Dataset):
    def __init__(self, poems, word2idx):
        self.data = []
        for poem in poems:
            words = poem.split()
            indices = [word2idx[word] for word in words if word in word2idx]
            # Create input-target pairs
            for i in range(1, len(indices)):
                self.data.append((indices[:i], indices[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(data_loader):
            batch_size = input_seq.size(0)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device).view(-1)

            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)

            optimizer.zero_grad()

            # Forward pass
            output, hidden = model(input_seq, hidden)

            # Compute loss
            loss = criterion(output, target_seq)

            # Backward pass and optimization
            loss.backward()

            # (Optional) Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")


# Example generation function (placeholder)
def generate_haiku(model, word2idx, idx2word, prompt):
    # Implement your haiku generation logic here
    return "Generated haiku based on the prompt."


if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = 'haikus.txt'

    poems = preprocess_poems(data_path)

    # Build vocabulary
    word2idx = build_vocab(poems)
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Create dataset and dataloader
    dataset = PoemDataset(poems, word2idx)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Increased batch_size for efficiency

    # Hyperparameters
    vocab_size = len(word2idx)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_epochs = 50
    learning_rate = 0.001

    # Initialize the model, criterion, and optimizer
    model = HaikuGRU(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, data_loader, criterion, optimizer, num_epochs, device)

    # Generate a haiku
    prompt = "Whispering winds"  # Replace this with your prompt
    generated_haiku = generate_haiku(model, word2idx, idx2word, prompt)
    print("\nGenerated Haiku:")
    print(generated_haiku)

