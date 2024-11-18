import re

def clean_data(file_path):
    with open(file_path, 'r') as f, open(file_path + 'cleaned', 'w') as f_out:

        for line in f:
            cleaned_line = re.sub(r'[^a-zA-Z\s]', '', line)
            # Write the cleaned line to the output file
            f_out.write(cleaned_line)

clean_data('poems.txt')