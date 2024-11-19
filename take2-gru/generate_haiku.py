import torch
from count_syllables import count_syllables

def generate(model, prompt, syllable_dictionary, word2idx, idx2word, device):
    """

    """

    model.eval()

    words = prompt.split()

    prompt_seq = []
    for word in words:
        prompt_seq.append(word2idx.get(word, word2idx['<unk>']))

    # TODO: change prompt_seq to a tensor
    hidden = None

    # generate the text, and split based on lines

    with torch.no_grad():
        # TODO: loop over the correct number of syllables
        # TODO: re-prompt the model if it doesn't generate an output with the correct number of syllables
        output, hidden = model(prompt_seq, hidden)
        # TODO: use softmax to get a probability distribution from the output logits
        # TODO: use torch.multinomial to get the next word
        # TODO: convert the next word (it is an index) to an actual word


    return