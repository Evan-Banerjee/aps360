import re
import os

def clean_data(file_path):
    file_name, file_root = os.path.splitext(file_path)

    pattern = r'[\u00A0\u2000-\u200A\u2028\u2029\u205F\u3000\uFEFF]'

    with open(file_path, 'r', encoding='utf-8') as f, open(file_name + '-cleaned-' + file_path, 'w') as f_out:
        previous_line = ''
        #i = 0
        for line in f:
            # regex and replace
            line = re.sub(r"[^a-zA-Z\s]", '', line)  # keep whitespace

            line = line.replace('\r', '') # remove all carriage returns
            line = line.replace('\n\n', '') # remove double newlines

            line = re.sub(pattern, ' ', line)  # any non-space Unicode white space

            line = re.sub(r'[ \t]+\n', '\n', line)  # removes spaces or tabs that appear immediately before a newline
            line = re.sub(r'\n[ \t]+', '\n', line)  # replace newlines followed by spaces or tabs with a single newline

            line = re.sub(r'[ \t]+', ' ', line)  # replaces one or more spaces/tabs with a single space
            line = re.sub(r'^[ \t]+|[ \t]+$', '', line)  # removes spaces or tabs at the beginning or end of the string

            line = re.sub(r'-{2,}', '-', line) # remove duplicate hyphens and replace with a single hyphen
            line = re.sub(r'-', ' ', line) # remove hyphenations
            line = re.sub(r' {2,}', ' ', line) # remove two or more consecutive spaces

            cleaned_line = line
            #cleaned_line = re.sub(r"[^a-zA-Z\s']", '', line)

            # Write the cleaned line to the output file
            if cleaned_line.isspace():

                if not previous_line.isspace():
                    f_out.write(cleaned_line)

            else:
                f_out.write(cleaned_line)

            #i += 1
            previous_line = line

clean_data('freakytext.txt')