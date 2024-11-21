

def create_dataset(cleaned_poems):
    """
    Creates a list of poems. Each poem is an element in the list.
    Appends a token to signify the end of a poem.
    :param cleaned_poems: Poems stripped of unnecessary characters
    :return: Poems stored in a list
    """

    poems = []

    with open(cleaned_poems, 'r', encoding='utf8') as f:
        current_poem = []
        for line in f:
            poem_line = line.strip()
            poem_line = poem_line.lower()
            if line.isspace():
                poems.append(' '.join(current_poem) + ' <eos>')
                current_poem = []
            else:
                current_poem.append(poem_line)

    return poems