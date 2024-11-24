def create_dataset(cleaned_poems, all_lines, max_lines):
    """
    Creates a list of poems. Each poem is an element in the list.
    Appends a token to signify the end of a poem.
    :param cleaned_poems: Poems stripped of unnecessary characters
    :return: Poems stored in a list
    """

    poems = []

    with open(cleaned_poems, 'r', encoding='utf8') as f:
        current_poem = []
        i = 0
        for line in f:
            if not all_lines:
                if i > max_lines:
                    break
            poem_line = line.strip()
            poem_line = poem_line.lower()
            if poem_line == '' and not line == '\n':
                for char in line:
                    print(f"Character: {repr(char)}, Unicode Code: U+{ord(char):04X}")
                print('Blank line')
            if line.isspace():
                poems.append(' '.join(current_poem))
                current_poem = []
            else:
                current_poem.append(poem_line)
            i += 1

    return poems