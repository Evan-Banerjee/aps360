# utils/read_haikus.py

def read_haikus_from_file(file_path):
    """
    Reads haikus from a text file.
    Assumes each haiku consists of three lines separated by a blank line.
    Returns a list of haikus where each haiku is a single string with lines concatenated.

    Args:
        file_path (str): Path to the haikus text file.

    Returns:
        list of str: List containing each haiku as a single string.
    """
    haikus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        haiku = []
        for line in f:
            stripped = line.strip()
            if stripped:
                haiku.append(stripped)
            else:
                if haiku:
                    # Join the three lines into a single string and append <eos>
                    haikus.append(' '.join(haiku) + ' <eos>')
                    haiku = []
        # Add the last haiku if file doesn't end with a blank line
        if haiku:
            haikus.append(' '.join(haiku) + ' <eos>')
    return haikus