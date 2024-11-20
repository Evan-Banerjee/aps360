import re
import os

def clean_data(file_path):
    file_name, file_root = os.path.splitext(file_path)

    with open(file_path, 'r', encoding='utf-8') as f, open(file_name + '-cleaned-' + file_path, 'w') as f_out:

        for line in f:
            line = re.sub(r'-', ' ', line)
            cleaned_line = re.sub(r"[^a-zA-Z\s']", '', line)
            # Write the cleaned line to the output file
            f_out.write(cleaned_line)

clean_data('morepoems.txt')