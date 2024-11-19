from create_dataset import create_dataset
from make_dictionary import make_dictionary
from poem_data import PoemDataset

def main():
    poems_path = 'data/poems-cleaned-poems.txt'
    poems = create_dataset(poems_path)

    word2idx, idx2word = make_dictionary(poems)
    assert len(word2idx) == len(idx2word)
    print(f'Length of dictionary: {len(word2idx)} words')

    poem_dataset = PoemDataset(poems, word2idx)
    print(poem_dataset[0])

    print('stop')

    #torchinfo.summary(MyModel(), (512, 10, 3))

if __name__ == '__main__':
    main()