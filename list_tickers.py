import os

def save_files_to_text(directory, output_file):
    """
    Saves the list of filenames in the given directory to a text file.
    
    Args:
    - directory (str): The path to the directory.
    - output_file (str): The path to the output text file.
    """
    with open(output_file, 'w') as out_file:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                out_file.write(filename + '\n')

# Example usage:
directory_path = 'C:/Users/ttbon/Desktop/stocks/data/Data/ETFs'
output_file_path = 'stocks_etfs.txt'
save_files_to_text(directory_path, output_file_path)