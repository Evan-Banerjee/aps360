# generate.py

import torch
from utils.count_syllables import count_syllables


def generate_haiku(model, word2idx, idx2word, prompt, max_length=50):
    """
    Generates a haiku based on the prompt, adhering to the 5-7-5 syllable structure.

    Args:
        model (nn.Module): Trained model.
        word2idx (dict): Mapping from word to index.
        idx2word (dict): Mapping from index to word.
        prompt (str): Starting prompt for the haiku.
        max_length (int): Maximum number of words to generate.

    Returns:
        str: Generated haiku.
    """
    model.eval()
    device = next(model.parameters()).device
    words = prompt.lower().split()
    input_seq = [word2idx.get(word, word2idx['<unk>']) for word in words]
    input_seq = torch.tensor([input_seq], dtype=torch.long).to(device)
    hidden = None

    generated = words.copy()
    lines = []
    line = generated.copy()
    syllables = sum([count_syllables(w) for w in line])
    target_syllables = [5, 7, 5]
    line_idx = 0

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]  # Get the last time step
            probabilities = torch.softmax(output, dim=1).squeeze().cpu()
            top_idx = torch.multinomial(probabilities, 1).item()
            predicted_word = idx2word.get(top_idx, '<unk>')

            if predicted_word == '<eos>':
                break

            line.append(predicted_word)
            syllables += count_syllables(predicted_word)
            generated.append(predicted_word)

            # Check if the current line has met the syllable requirement
            if syllables >= target_syllables[line_idx]:
                lines.append(' '.join(line))
                line = []
                syllables = 0
                line_idx += 1
                if line_idx >= 3:
                    break

            # Prepare the next input
            input_seq = torch.tensor([[top_idx]], dtype=torch.long).to(device)

    # If not all lines are completed, append what has been generated
    while line_idx < 3:
        if line:
            lines.append(' '.join(line))
            line = []
            syllables = 0
            line_idx += 1
        else:
            lines.append('')
            line_idx += 1

    return '\n'.join(lines)