import re

def remove_non_alphabet(input_file, output_file):
    # Open the input file for reading and output file for writing
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        # Process each line in the input file
        for line in infile:
            # Remove all non-alphabet characters (retain only a-z and A-Z)
            cleaned_line = re.sub(r'[^a-zA-Z\s]', '', line)
            # Write the cleaned line to the output file
            outfile.write(cleaned_line)

remove_non_alphabet("haikus2.txt", "haikus2-cleaned.txt")