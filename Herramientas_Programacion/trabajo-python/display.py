import os
from repository import Repository
from dna import DNA

class Display:

    def __init__(self):
        self.repository = Repository()
        self.dna = DNA()

    def ask_saving_DNA(self):
        ''' Helper method for asking where to save a .dna file
        Return
        -------
        bool
            False if file could not be saved
            True if file was saved successfully
        '''
        directories = self.repository.obtain_directories()
        print('Existing directories:')
        for directory in directories:
            print(f'> {directory}')

        directory_choice = input('Enter the directory where you want to save the file (press Enter to save in the current directory): ').strip()

        print('Saving DNA chain...')
        file_path = self.repository.save_dna_chain(self.dna, directory_choice)

        if file_path:
            print(f'Saved successfully at {file_path}.')
            return True
        else:
            print('An error has occurred. Please try again.')
            return False

    def validate_user_number(self, message):
        '''Validates that the user input is a numerical value
        Input Parameter
        --------------
        message: str
            Message to display when asking the user for a number.
        Return
        ------
        numerical_input: int
            User's input transformed into an integer
        None
            If user's input is no integer
        '''
        try:
            numerical_input = int(input(message))
            return numerical_input
        except ValueError:
            print('Number not detected. Please try again.')
            return

    def display_dna_files(self):
        '''Helper method to display the DNA files
         Return
         ------
         success: bool
            True when there are files that can be listed,
            False when there are no files with extension .dna
        '''
        success, lines = self.repository.list_dna_files()
        if success:
            for line in lines:
                print(line)
        return success

    def check_loaded_dna(self):
        ''' Helper method to check if there is a loaded DNA chain
        Return
        ------
        bool
            True if there is a DNA chain
            False when no DNA chain has been loaded or generated
        '''
        if not self.dna.dna_chain:
            print('First load or generate a DNA chain before using this option.')
            return False
        return True

    def first_menu(self):
        ''' Initialization menu for creating a DNA sequence, persistence, accessing the second menu and exiting the program.
        '''
        while True:
            print('------------------------- \n'
                  'DNA-Toolkit v0.1 \n'
                  'Please, select an option: \n'
                  '1 - Create new DNA chain \n'
                  '2 - Save DNA chain \n'
                  '3 - Load DNA from disk \n'
                  '4 - List all DNA info \n'
                  '5 - Delete DNA info \n'
                  '6 - Operations with DNA \n'
                  '7 - Exit')
            if not self.dna.dna_chain:
                print('[[There is no DNA chain loaded]]')
            else:
                print(f'Loaded chain: {self.dna.dna_chain}')
            option = self.validate_user_number('Please, enter a number: ')
            try:
                if option is None:
                    pass
                
                if 1 <= option <= 7:
                    print(f'You have selected option {option}')            

                    if option == 1:
                        print('Creating DNA chain...')
                        length_chain = self.validate_user_number('Please, provide the length of the DNA chain: ')
                        if length_chain is not None:
                            self.dna = DNA() #create new DNA object with default properties
                            self.dna.dna_chain = self.dna.create_dna_chain(length_chain)
                            print(f'Generated chain: {self.dna.dna_chain}')

                    elif option == 2:
                        if self.check_loaded_dna():
                            self.ask_saving_DNA()

                    elif option == 3:
                        if self.display_dna_files():
                            choice = self.validate_user_number('Which file do you want to load? Enter a number: ')
                            if choice is not None:
                                dna_object = self.repository.load_dna_file(choice)
                                if dna_object is not None and dna_object.dna_chain == '':
                                    print('File does not exist. Please enter a valid number.')
                                else:
                                    self.dna = dna_object
                                    print(f'Loaded chain: {self.dna.dna_chain}')
                        else:
                            print('There are no files to be loaded.')

                    elif option == 4:
                        if not self.display_dna_files():
                            print('There are no files to list.')

                    elif option == 5:
                        if self.display_dna_files():
                            choice = self.validate_user_number('Which file do you want to delete? Enter a number: ')
                            if choice is not None:
                                file_deleted = self.repository.delete_dna_file(choice)
                                if file_deleted:
                                    print('File deleted.')
                                else:
                                    print('File does not exist. Please enter a valid number.')
                        else:
                            print('There are no files to be deleted.')

                    elif option == 6:
                        if self.check_loaded_dna():
                            self.second_menu()

                    elif option == 7:
                        if self.dna.dna_chain != '' and not self.dna.saved:
                            save_choice = input(
                                'The current DNA chain is not saved. Do you want to save it? (y/n): ').lower()
                            if save_choice == 'y':
                                if self.ask_saving_DNA():
                                    print('Program exited.')
                                    break
                            elif save_choice == 'n':
                                print('Exiting without saving.')
                                break
                            else:
                                print('Answer not detected. Please try again.')
                        else:
                            print('Program exited.')
                            break
                else:
                    print('Option not found - please try again.')
            except:
                print('Unexpected error - please try again.')

    def second_menu(self):
        ''' DNA operations menu for regenerating, validating, and mutating DNA chain,
        measure frequencies and GC content, count appearances of a subsequence, synthesize reverse and complementary strands
        and going back to the first menu.
        '''
        while True:
            print('------------------------- \n'
                  'DNA-Toolkit v0.1 \n'
                  'Operations with DNA chain: \n'
                  '1 - Re-generate DNA chain \n'
                  '2 - Validate DNA chain \n'
                  '3 - Mutate DNA chain \n'
                  '4 - Measure frequencies \n'
                  '5 - Count subsequences \n'
                  '6 - Synthesize the reverse and complementary DNA strands \n'
                  '7 - Measure %GC \n'
                  '8 - Back')
            if not self.dna.dna_chain:
                print('[[There is no DNA chain loaded]]')
            else:
                print(f'Loaded chain: {self.dna.dna_chain}')
            option = self.validate_user_number('Please, insert an option: ')
            try:
                if option is None:
                        pass
                
                if 1 <= option <= 8:
                    print(f'You have selected option {option}')
           
                    if option == 1:
                        length = self.validate_user_number('Please provide the length of the DNA chain: ')
                        self.dna.dna_chain = self.dna.create_dna_chain(length)

                    elif option == 2:
                        if self.dna.validate_chain():
                            print('The DNA chain is valid.')
                        else:
                            print('The DNA chain is not valid.')

                    elif option == 3:
                        if self.dna.validate_chain():
                            number_mutations = self.validate_user_number('Please provide the number of mutations: ')
                            if number_mutations != None:
                                previous_chain = self.dna.dna_chain
                                mutation_output = self.dna.mutate_chain(number_mutations)
                                if mutation_output == None:
                                    print('Number out of range - not enough nucleotides to mutate')
                                else:
                                    print(f'Previous DNA chain: {previous_chain}')
                                    print(f'Mutated DNA chain: {self.dna.dna_chain}')
                        else:
                            print('Please provide a valid DNA chain first.')

                    elif option == 4:
                        if self.dna.validate_chain():
                            frequencies = self.dna.measure_frequencies()
                            print(f'Nucleotide frequencies {frequencies}')
                        else:
                            print('Nucleotide frequencies could not be calculated due to unknown nucleotide.')

                    elif option == 5:
                        if self.dna.validate_chain():
                            subsequence = input('Please provide the subsequence: ').upper()
                            nucleotide_validation = True
                            for character in subsequence:
                                if character not in "ATCG":
                                    nucleotide_validation = False
                                    break
                            if nucleotide_validation:
                                number_subsequence = self.dna.count_subsequence(subsequence)
                                print(f'{subsequence} appears {number_subsequence} times in {self.dna.dna_chain}.')
                            else:
                                print('The subsequence has unknown nucleotides.')
                        else:
                            print('Please provide a valid DNA chain first.')

                    elif option == 6:
                        if self.dna.validate_chain():
                            complementary, reverse = self.dna.synthesize_complementary_reverse()
                            print(f"Synthesizing complementary and reverse chains... \n"
                                f"The DNA chain is: 3' {self.dna.dna_chain} 5' \n"
                                f"The complementary chain is: 5' {complementary} 3' \n"
                                f"The reverse chain is: 5' {reverse} 3'")
                        else:
                            print('Please provide a valid DNA chain first.')

                    elif option == 7:
                        if self.dna.validate_chain():
                            gc_percent = self.dna.measure_gc_content()
                            print(f'The GC% is: {gc_percent:.2f} %')
                        else:
                            print('Please provide a valid DNA chain first.')

                    elif option == 8:
                        break
                else:
                    print('Option not found - please try again.')
            except:
                print('Unexpected error - please try again.')


if __name__ == "__main__":
    my_display = Display()
    my_display.first_menu()
