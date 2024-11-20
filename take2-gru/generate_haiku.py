import torch
from count_syllables import count_syllables

def generate(model, prompt, syllable_dictionary, word2idx, idx2word, device):
    """
    Create a haiku with the model
    :param model:
    :param prompt:
    :param syllable_dictionary:
    :param word2idx:
    :param idx2word:
    :param device:
    :return: A haiku as a list of lines, each line has the words for that syllable
    """

    #model.eval()

    words = prompt.split()

    prompt_seq = []
    for word in words:
        prompt_seq.append(word2idx.get(word, word2idx['<unk>']))

    # TODO: change prompt_seq to a tensor
    prompt_seq_tensor = torch.tensor(data=prompt_seq, dtype=torch.long, device=device) # should this be a floating-point tensor or does the model return integers?

    hidden = None

    target_syllables = [5,7,5]

    # generate the text, and split based on lines

    # final haiku stored here
    haiku = []

    with torch.no_grad():
        # TODO: loop over the correct number of syllables
        # Total: 17 syllables
        # First line is 5, 2nd is 7, 3rd is 5

        for syllable in target_syllables:
            current_syllables = 0

            line_is_done = False

            # stores the current line
            line = []

            # generate a new word

            while not line_is_done:

                output, hidden = model(prompt_seq_tensor, hidden)

                normalized_output = torch.softmax(output, dim=-1).squeeze().cpu()

                next_index = torch.multinomial(normalized_output, 1).item()

                next_word = idx2word.get(next_index, '<unk>')

                # check the number of syllables that were generated

                if next_word == '<eos>': # maybe we shouldn't do this, and force the model to just move on when we decide it has met the syllables
                    break
                else:
                    add_syllables = count_syllables(word=next_word, syllable_dictionary=syllable_dictionary)

                    if (add_syllables + current_syllables) <= syllable:
                        line.append(next_word)
                        current_syllables += add_syllables
                        prompt_seq.append(next_index)
                        prompt_seq_tensor = torch.tensor(data=prompt_seq, dtype=torch.long, device=device)


                    if (add_syllables + current_syllables) == syllable:
                        line_is_done = True

            haiku.append(line)







        # TODO: re-prompt the model if it doesn't generate an output with the correct number of syllables

        # TODO: use softmax to get a probability distribution from the output logits
        # TODO: use torch.multinomial to get the next word
        # TODO: convert the next word (it is an index) to an actual word


    return