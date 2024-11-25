import matplotlib.pyplot as plt
import nltk
from nltk.corpus import cmudict

from count_syllables import count_syllables


def plot_syllable_counts(test_haikus_file, plot_file):
    syllable_dictionary = cmudict.dict()

    syllable_counts = []
    line_count = 0
    with open(test_haikus_file, 'r') as f:
        for line in f:
            if line.isspace():
                syllable_counts.append(line_count)
                line_count = 0
            else:
                for word in line.split():
                    line_count += count_syllables(word, syllable_dictionary)

    plt.figure(figsize=(10, 6))
    plt.hist(syllable_counts, bins=range(min(syllable_counts), max(syllable_counts) + 1), edgecolor='black', alpha=0.7)
    plt.xticks(range(min(syllable_counts), max(syllable_counts) + 1))
    # Add labels and title
    plt.title("Syllables per poem vs. Frequency")
    plt.xlabel("Syllables per poem")
    plt.ylabel("Frequency")

    # Display the plot
    plt.grid(True)
    plt.savefig(plot_file, format="png", dpi=300)
    plt.show()


plot_syllable_counts('data/fuzz_output_last_word.txt', 'syllable_count_last_word.png')