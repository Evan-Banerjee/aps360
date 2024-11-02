import csv

# Open the CSV file and the output text file
with open("haikus_unique.csv", "r") as csv_file, open("haikus2.txt", "w") as txt_file:
    reader = csv.DictReader(csv_file, delimiter="\t")
    # Loop through each row in the CSV file
    for row in reader:
        haiku = row["haiku"]

        # Split the haiku by "|" and write each part on a new line
        haiku_lines = haiku.split("|")
        txt_file.write("\n".join(haiku_lines))

        # Add an extra newline between haikus
        txt_file.write("\n\n")