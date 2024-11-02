# utils/count_syllables.py

import nltk
from nltk.corpus import cmudict

# Ensure CMU Pronouncing Dictionary is downloaded
nltk.download('cmudict', quiet=True)

# Load the CMU Pronouncing Dictionary
d = cmudict.dict()

def count_syllables(word):
    """
    Counts the number of syllables in a word using the CMU Pronouncing Dictionary.

    Args:
        word (str): The word to count syllables for.

    Returns:
        int: Number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        # Take the first pronunciation
        syllable_counts = [len([y for y in x if y[-1].isdigit()]) for x in d[word]]
        return syllable_counts[0]
    else:
        # Assume 1 syllable if word not found
        return 1