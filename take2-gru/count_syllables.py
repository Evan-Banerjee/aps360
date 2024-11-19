import syllapy

def count_syllables(word, syllable_dictionary):
    """
    Count the number of syllables in a word.
    :param word: The word to count syllables for
    :param syllable_dictionary: CMU dictionary of syllable counts
    :return: The number of syllables in the word
    """

    # check if it is in the cmudict, if not count it algorithmically
    phones = syllable_dictionary.get(word)

    if phones:
        phones0 = phones[0]
        count = len([p for p in phones0 if p[-1].isdigit()])
        return count
    else:
        count = syllapy.count(word)
        return count