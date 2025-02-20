from client import Client
from FileHandler import FileHandler
import re

class Display:

    def __init__(self):
        self.client = Client()
        self.filehandler = FileHandler()
        self.species_of_interest = []
        self.species_information = []
        self.id_list = []
        self.sequences_list = []
        self.sequences_from_species = {}

    def start_menu(self):
        """Starting menu for the program. Calls helper functions for clarity and a better overview of each function.
        """
        while True:
            print("\n===== MENU =====")
            print("1. Search for species of interest")
            print("2. Read file with sequence IDs")
            print("3. Fetch sequences from the database")
            print("4. Check sequences by species of interest")
            print("5. Find subsequences in sequences")
            print("6. Save sequences of species of interest")
            print("7. Exit")
            choice = input("Select an option (1-7): ")

            if choice == "1":
                self.menu_search_species()
            elif choice == "2":
                self.menu_read_file()
            elif choice == "3":
                self.menu_fetch_sequences()
            elif choice == "4":
                self.menu_check_sequences_by_species()
            elif choice == "5":
                self.menu_find_subsequence()
            elif choice == "6":
                self.menu_save_sequences()
            elif choice == "7":
                print("Exiting the program.")
                break
            else:
                print("Invalid option. Please try again.")

    def menu_search_species(self):
        """Helper method to search for species in the database and filter for a species of interest
        """
        try:
            species_input = input("Enter scientific names separated by commas (Homo sapiens default): ").lower()
            self.species_of_interest = [species.strip() for species in species_input.split(",")]

            if len(self.species_of_interest) == 1 and self.species_of_interest[0] == '':
                self.species_of_interest = ["homo sapiens"]

            self.species_information = self.client.obtain_species_from_database(self.species_of_interest)

            if len(self.species_information) > 0:
                print(f"Found species: {[species.display_name for species in self.species_information]}")
            else:
                print("No species found for the provided names.")

        except Exception as e:
            print(f"Error while searching for species: {e}")

    def menu_read_file(self):
        """Helper method to read Ensembl IDs from a file in the same directory
        """
        while True:
            try:
                input_file = input("Enter the name of the file with sequence IDs: ")
                self.id_list, invalid_ids = self.filehandler.extract_id_from_file(input_file)

                if self.id_list:
                    print(f"Successfully read valid IDs: {self.id_list}")

                if len(invalid_ids) > 0:
                    print(f"The following IDs are invalid: {invalid_ids}")
                break

            except FileNotFoundError:
                print("File not found. Please try again.")
            except ValueError as e:
                print(f"Error: {e}")

    def menu_fetch_sequences(self):
        """Helper method to search for sequences in the database from the IDs previously read
        """
        try:
            if len(self.id_list) == 0:
                print("You need to read a file with IDs first (option 2).")
                return

            self.sequences_list, missing_ids = self.client.obtain_sequence_from_database(self.id_list)
            print(f"Successfully fetched {len(self.sequences_list)} sequences from the database.")

            if missing_ids:
                print(f"The following IDs could not be found in the database: {missing_ids}")

        except Exception as e:
            print(f"Error while fetching sequences: {e}")

    def menu_check_sequences_by_species(self):
        """Helper method to look for fetched sequences of a given species of interest
        """
        try:
            if not self.species_information or len(self.id_list) == 0:
                print("You need to search for species and fetch sequences first (options 1 and 3).")
                return

            self.sequences_from_species = self.client.obtain_species_from_sequence(
                self.sequences_list, self.species_information
            )
            print(f"Checked sequences for species of interest. Results: {self.sequences_from_species}")

        except Exception as e:
            print(f"Error while checking sequences by species: {e}")

    def menu_find_subsequence(self):
        """Helper method to find a subsequence in the fetched sequences
        """
        try:
            if len(self.sequences_list) == 0:
                print("You need to fetch sequences first (option 3).")
                return

            while True:
                subsequence = input("Enter the subsequence to search for (TTTT default): ").upper()
                if bool(re.match(r'^[0-9]+$', subsequence)):
                    print("Subsequence cannot contain numbers.")
                else:
                    break

            if subsequence == '':
                subsequence = "TTTT"

            subsequence_positions = self.client.subsequence_positions(self.sequences_list, subsequence)
            print(f"Subsequences of length {len(subsequence)}")

            for seq_id, positions in subsequence_positions.items():
                print(f"Sequence ID: {seq_id}")
                print(f"{len(positions)} found")
                for item in positions:
                    print(f"POS: {item}")

        except Exception as e:
            print(f"Error while finding subsequence: {e}")

    def menu_save_sequences(self):
        """Helper method to save sequences of the given species of interest in a file
        """
        try:
            if len(self.sequences_from_species) == 0 or len(self.sequences_list) == 0:
                print("You need to check sequences by species first (option 4).")
                return

            success = self.filehandler.save_sequences_to_file(self.sequences_from_species, self.sequences_list)

            if success:
                print("Sequences of species of interest saved successfully.")
            else:
                print("Error while saving sequences.")

        except Exception as e:
            print(f"Error while saving sequences: {e}")

if __name__ == "__main__":
    display = Display()
    display.start_menu()