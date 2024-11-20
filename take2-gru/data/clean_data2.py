import os

def clean_more_text(text_file, output_file):

    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    consecutive_blank_lines = 0

    for line in lines:
        stripped_line = line.strip()

        if stripped_line:  # Line contains text
            if consecutive_blank_lines >= 2 and processed_lines and processed_lines[-1] != '\n':
                # Add exactly two blank lines between blocks
                processed_lines.append('\n')
                processed_lines.append('\n')
            elif consecutive_blank_lines > 0 and processed_lines and processed_lines[-1] == '\n':
                # Ensure no trailing multiple blank lines
                pass

            processed_lines.append(stripped_line + '\n')  # Add the text line
            consecutive_blank_lines = 0  # Reset blank line counter
        else:  # Line is blank
            consecutive_blank_lines += 1

        # Write the processed lines to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(processed_lines)


clean_more_text('morepoems.txt', 'more_cleaned_poems.txt')