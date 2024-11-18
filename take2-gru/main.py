from create_dataset import create_dataset
from make_dictionary import make_dictionary

def main():
    poems_path = 'data/poemscleaned.txt'
    poems = create_dataset(poems_path)

    word2idx, idx2word = make_dictionary(poems)
    assert len(word2idx) == len(idx2word)
    print(f'Length of dictionary: {len(word2idx)} words')

    

    #torchinfo.summary(MyModel(), (512, 10, 3))

if __name__ == '__main__':
    main()