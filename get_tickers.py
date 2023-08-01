import os

# Specify the directory you want to scan for files
directory = 'Data/ETFs'

# Get a list of all files in the directory
files = os.listdir(directory)

# Open a new text file and write the file names
with open('file_names.txt', 'w') as f:
    for file in files:
        f.write(file + '\n')