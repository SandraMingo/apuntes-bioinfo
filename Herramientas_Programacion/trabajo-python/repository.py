import glob
import os
from dna import DNA
from datetime import datetime

class Repository:

    def __init__(self):
        self.directory_files = {}

    def save_dna_chain(self, dna_object, directory=''):
        '''Saves the given DNA object into a file with extension .dna
        Input Parameter
        --------------
        dna_object: object
            DNA object to be saved in a file on disk
        directory: str (optional)
            The directory where the file should be saved.
            Default empty: saves in the current directory.
        Returns
        -------
        file_path: str
            Path of the saved file if successful.
        False: bool
            If the save process failed.
        '''
        try:
            if directory == '':
                if dna_object.directory and os.path.isfile(dna_object.directory):
                    directory, filename = os.path.split(dna_object.directory)
                else:
                    directory = os.getcwd()
                    filename = f'DNA_{datetime.now().strftime("%d%m%Y%H%M%S")}.dna'
            else:
                filename = f'DNA_{datetime.now().strftime("%d%m%Y%H%M%S")}.dna'

            if not os.path.exists(directory):
                os.makedirs(directory)

            file_path = os.path.join(directory, filename)
            with open(file_path, 'w') as file:
                file.write(dna_object.dna_chain)

            # Mark DNA as saved and update the full path (directory + filename)
            dna_object.saved = True
            dna_object.directory = file_path

            return os.path.relpath(file_path)

        except:
            return False

    def list_dna_files(self):
        '''Displays all files with extension .dna, their sequence, length and nucleotide frequencies.
        Returns
        -------
        success: bool
            Indicates if files were listed successfully.
        files: list
            A list of formatted strings representing each file's details (number, name with relative path,
            length, sequence and nucleotide frequencies).
        '''
        self.directory_files = {}
        file_list = glob.glob('**/*.dna', recursive = True)

        if len(file_list) == 0:
            return False, []
        else:
            files = []
            counter = 1
            for file in file_list:
                with open(file, 'r') as current_file:
                    content = current_file.read()
                    dna_object = DNA()
                    dna_object.dna_chain = content
                    self.directory_files[counter] = file
                    file_info = f'{counter} - {file} - Length: {len(content)}, Sequence: {content}, Frequencies: {dna_object.measure_frequencies()}'
                    files.append(file_info)
                    counter += 1
            return True, files

    def load_dna_file(self, choice):
        '''Load a DNA sequence on a file on disk into the program as a DNA object
        Input Parameter
        --------------
        choice: int
            Number of the file to be loaded
        Return
        -----
        dna_object: DNA
            If file exists, DNA object loaded from the file
            If file does not exist, empty DNA object
        '''
        try:
            dna_object = DNA()

            file_path = self.directory_files.get(choice, None)
    
            if not file_path:
                return dna_object
            
            with open(file_path, 'r') as loaded_file:
                dna_chain = loaded_file.read()
                dna_object.dna_chain = dna_chain.upper()
                dna_object.directory = os.path.dirname(file_path)
                dna_object.saved = True
                return dna_object

        except KeyError:
            return dna_object

    def delete_dna_file(self, choice):
        '''Delete a file with extension .dna from the disk
        Input Parameter
        --------------
        choice: int
            Number of the file to be deleted
        Return
        ------
        True: bool
            When file removed
        False: bool
            When file for removing does not exist; choice exceeds the number of existing files
        '''
        try:
            file = self.directory_files.get(choice, None)
            if not file:
                return False

            os.remove(file)
            return True
        except KeyError:
            return False

    def obtain_directories(self):
        ''' Helper method to obtain a list with the subdirectories in the current working directory
        Return
        ------
        directories: list
            List with all the directories inside the current working directory
        '''
        current_directory = os.getcwd()
        directories = [directory for directory in os.listdir(current_directory) if
                       os.path.isdir(os.path.join(current_directory, directory))]
        return directories