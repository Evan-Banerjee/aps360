import torch
from count_syllables import count_syllables
from grammar import GrammarChecker

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

    model.eval()

    words = prompt.split()

    prompt_seq = []

    line = []

    for word in words:
        prompt_seq.append(word2idx.get(word, word2idx['<unk>']))
        line.append(word)

    prompt_seq_tensor = torch.tensor(data=[prompt_seq], dtype=torch.long).to(device) # should this be a floating-point tensor or does the model return integers?

    hidden = None

    target_syllables = [5,7,5]

    start_syllables = count_syllables(prompt, syllable_dictionary)

    current_syllables = start_syllables

    # generate the text, and split based on lines

    # final haiku stored here
    haiku = []

    checker = GrammarChecker()

    with torch.no_grad():
        # Total: 17 syllables
        # First line is 5, 2nd is 7, 3rd is 5

        for syllable in target_syllables:


            line_is_done = False

            # stores the current line


            # generate a new word

            while not line_is_done:

                output, hidden = model(prompt_seq_tensor, hidden)

                output = output[:, -1, :] # remove the batch size from the tensor to leave us with just the word logit tensors, and the vocabulary

                normalized_output = torch.softmax(output, dim=1).squeeze().cpu() # normalize the logits into probabilities for each row, where there is one distribution per word. Squeeze is there to remove and size 1 dimensions if they exist

                next_index = torch.multinomial(normalized_output, num_samples=1).item()

                next_word = idx2word.get(next_index, '<unk>')

                # check the number of syllables that were generated

                if next_word == '<eos>': # maybe we shouldn't do this, and force the model to just move on when we decide it has met the syllables
                    line_is_done = True
                else:

                    if next_word != '<unk>':

                        add_syllables = count_syllables(word=next_word, syllable_dictionary=syllable_dictionary)

                        if (add_syllables + current_syllables) <= syllable:
                            line.append(next_word)
                            current_syllables += add_syllables
                            prompt_seq.append(next_index)
                            prompt_seq_tensor = torch.tensor(data=[prompt_seq], dtype=torch.long, device=device)


                        if (current_syllables) >= syllable:
                            line_is_done = True
                
                if line_is_done:
                    #start checking for grammatical correctness
                    sentence = " ".join(line)
                    print(f"total number of syllables (should be {syllable}) for '{sentence}' is {current_syllables}")
                    
                    checker_output = checker.check_grammar(sentence)
                    #print(f"output of checker is {checker_output}\n")
                    grammar_status = checker_output['status'] == "Correct"
                    print(f"line '{sentence}' has correct grammar: {grammar_status}")
                    if not grammar_status:
                        #check if line can be fixed
                        if checker_output['message'] != "The sentence appears nonsensical.":
                            #print(f"corrected line is {checker.correct_sentence(sentence)}")
                            print("restarting line\n")
                            prompt_seq = prompt_seq[:-len(line)]
                            line = []
                            current_syllables = 0
                            
                            if len(prompt_seq) == 0: #if we were on the first line, reset everything
                                for word in words:
                                    prompt_seq.append(word2idx.get(word, word2idx['<unk>']))
                                    line.append(word)
                                prompt_seq_tensor = torch.tensor(data=[prompt_seq], dtype=torch.long).to(device)
                                current_syllables = count_syllables(prompt, syllable_dictionary)
                            else:
                                prompt_seq_tensor = torch.tensor(data=[prompt_seq], dtype=torch.long, device=device)

                            line_is_done = False
                        else:
                            #print(f"corrected line is {checker.correct_sentence(sentence)}")
                            print("restarting line\n")
                            prompt_seq = prompt_seq[:-len(line)]
                            line = []
                            current_syllables = 0
                            
                            if len(prompt_seq) == 0: #if we were on the first line, reset everything
                                for word in words:
                                    prompt_seq.append(word2idx.get(word, word2idx['<unk>']))
                                    line.append(word)
                                prompt_seq_tensor = torch.tensor(data=[prompt_seq], dtype=torch.long).to(device)
                                current_syllables = count_syllables(prompt, syllable_dictionary)
                            else:
                                prompt_seq_tensor = torch.tensor(data=[prompt_seq], dtype=torch.long, device=device)

                            line_is_done = False




            haiku.append(line)
            line = []
            current_syllables = 0

    final_haiku = ''
    for line in haiku:
        temp_line = ' '.join(line)
        final_haiku += temp_line + '\n'

    final_haiku = final_haiku.rstrip('\n')

    return final_haiku