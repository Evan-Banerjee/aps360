from create_dataset import create_dataset
from make_dictionary import make_dictionary
from poem_data import PoemDataset
from model import HaikuGRU
from train import train
from generate_haiku import generate
from pad_data import CollateFn
#from header import start_menu

import nltk
from nltk.corpus import cmudict
#from torchinfo import summary
import torch


def main():
    # skip_training = start_menu

    selection = input("Skip training? (Y/N): ")
    if selection == 'Y':
        skip_training = True
    elif selection == 'N':
        skip_training = False
    else:
        skip_training = False

    #poems_path = 'data/poems-cleaned-poems.txt'
    # poems_path = 'data/morepoems2-cleaned-morepoems2.txt'
    # poems_path = 'data/haikus2-cleaned.txt'
    poems_path = 'data/all_poems-cleaned-all_poems.txt'
    #poems_path = 'data/freakytext-cleaned-freakytext.txt'

    all_lines = True
    max_lines = 100000

    poems = create_dataset(poems_path, all_lines, max_lines)

    for poem in poems:
        if len(poem) == 0:
            print(poem)

    nltk.download('cmudict', quiet=True)

    syllable_dictionary = cmudict.dict()

    word2idx, idx2word = make_dictionary(poems)
    assert len(word2idx) == len(idx2word)
    print(f'Length of dictionary: {len(word2idx)} words')
    vocab_size = len(idx2word)
    padding_idx = word2idx.get('<pad>', None)
    unknown_idx = word2idx.get('<unk>', None)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    if not skip_training:

        # Hyperparameters -----
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 2
        dropout = 0
        bidirectional = True

        model_parameters = [('vocab_size', vocab_size), ('embedding_dim', embedding_dim), ('padding_idx', padding_idx),
                            ('hidden_dim', hidden_dim), ('num_layers', num_layers), ('dropout', dropout),
                            ('bidirectional', bidirectional)]

        model = HaikuGRU(vocab_size=vocab_size,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=bidirectional)

        learning_rate = 1e-3
        batch_size = 64
        epochs = 40
        criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        grad_norm = 0
        clip_grad = False

        save_location = 'models'
        save_freq = 10  # every n epochs

        dataset = PoemDataset(poems, word2idx)
        collate_fn = CollateFn(padding_index=padding_idx)

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn,
                                                  num_workers=4, pin_memory=True)

        model = train(model=model,
                      data_loader=data_loader,
                      criterion=criterion,
                      optimizer=optimizer,
                      num_epochs=epochs,
                      learning_rate=learning_rate,
                      vocab_size=vocab_size,
                      grad_norm=grad_norm,
                      clip_grad=clip_grad,
                      save_location=save_location,
                      save_frequency=save_freq,
                      device=device,
                      model_params=model_parameters)

    else:
        # model = HaikuGRU()
        # model = torch.load('models/haiku_model_epoch_20.pth') # change this to a var
        # model.load_state_dict(torch.load('models/haiku_model_epoch_20.pth'))
        model_data = torch.load('models/haiku_model_epoch_50.pth', map_location=device)
        config = model_data['config']
        model = HaikuGRU(**config)
        model.to(device=device)
        model.load_state_dict(model_data['state_dict'])

    prompt = input(
        'Enter a prompt (type exit() to end the program): ')  # assuming a max of 5 syllables on the input to start, might change

    while prompt != 'exit()':
        haiku = generate(model=model,
                         prompt=prompt,
                         syllable_dictionary=syllable_dictionary,
                         word2idx=word2idx,
                         idx2word=idx2word,
                         device=device)

        print(haiku)

        prompt = input('Enter a prompt (type exit() to end the program): ')

if __name__ == '__main__':
    main()