import csv

# Define the header
header = ['alpha', 'qw', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']

# Path to the CSV file
file_path = "data.csv"

# Create the file and write the header
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)