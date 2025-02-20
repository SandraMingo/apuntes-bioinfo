class FileHandler:
    
    def extract_id_from_file(self, input_file):
        """ 
        Extract the IDs from a file.

        Input Parameters:
        ----------------
        input_file: str
            Name of the file that contains Ensembl IDs.
            It needs to be in the same directory as the script.

        Return:
        ------
        valid_ids: list of str
            List of valid Ensembl IDs
        invalid_ids: list of str
            List of invalid Ensembl IDs
        """
        try:
            with open(input_file, "r") as f:
                file_content = [line.strip() for line in f]

            #first filter of validation - all Ensembl IDs should start with ENS
            #alternatively: not check starts with ENS, pass it later to database and
            #included in missing_ids in client.obtain_sequence_from_database
            valid_ids = [line for line in file_content if line.startswith("ENS")]
            invalid_ids = [line for line in file_content if not line.startswith("ENS") and len(line) > 0]

            if not valid_ids:
                raise ValueError("No valid IDs found in the file.")

            return valid_ids, invalid_ids

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{input_file}' not found.")
        except ValueError:
            raise ValueError(f"File '{input_file}' is empty.")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")

    def save_sequences_to_file(self, sequence_from_species_dict, sequence_list):
        """Write a sequence into a file

        Parameters:
        -----------
        sequence_from_species_dict: dictionary
            sequence ID as key and True or False if it is a sequence from the species of interest
        sequence_list: list
            list containing the sequences

        Return:
        -------
        bool
            True if saved correctly, False if an error happened
        """
        try:
            sequence_dict = {sequence.seq_id: sequence for sequence in sequence_list}

            for identifier, is_valid in sequence_from_species_dict.items():
                if is_valid:
                    sequence = sequence_dict.get(identifier)
                    if sequence:
                        with open(f"{identifier}.txt", "w") as f:
                            f.write(sequence.seq)

            return True
        except:
            return False