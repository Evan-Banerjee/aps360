import csv
import os
import re

def extract_column_to_text(csv_file_path, output_text_file, number_to_extract, column_to_extract, delimiter=',', has_header=True):
    extracted_elements = []

    with open(csv_file_path, mode='r', encoding='utf-8', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)

        if has_header:
            header = next(csv_reader, None)

            if header is None:
                raise ValueError('Header not found in csv file')

        row_count = 0

        for row in csv_reader:
            text = row[column_to_extract]
            text = text.replace('\r', '') # carriage return
            text = text.replace('\n\n', '')
            text = re.sub(r'\u00A0', ' ', text) # non breaking space char
            text = re.sub(r'\u2009', ' ', text) # thin space
            text = re.sub(r'\u200A', ' ', text) # hair space
            text = re.sub(r'\u2028', ' ', text) # line seperator (not new line or carriage return)
            text = re.sub(r'\u2006', ' ', text) # Six-Per-Em Space
            text = re.sub(r'\n[ \t]+', '\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'-', ' ', text)
            text = re.sub(r"[^a-zA-Z\s']", '', text) # everything except for apostrophe

            extracted_elements.append(text)

            row_count += 1

            if row_count >= number_to_extract:
                break

    with open(output_text_file, mode='w', encoding='utf-8') as output_file:
        for element in extracted_elements:
            output_file.write(element)
            output_file.write('\n\n')


extract_column_to_text('kaggle_poem_dataset.csv', 'morepoems2.txt', 100000000000, 4)
