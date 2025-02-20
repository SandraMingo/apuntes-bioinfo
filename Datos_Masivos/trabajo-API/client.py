import requests, json
from sequence import Sequence
from species import Species

class Client:
    def __init__(self):
        self.server = "https://rest.ensembl.org"

    def obtain_species_from_database(self, species_of_interest_list):
        """
        Fetches species information from the Ensembl database based on a list of species names.

        Parameters:
        ----------
        species_of_interest_list : list of str
            List of species names to filter from the database.

        Returns:
        -------
        list
            A list of species objects filtered by the species of interest.
        """
        try:
            headers = {"Content-Type": "application/json"}
            species_endpoint = f"/info/species/"
            response = requests.get(f"{self.server}{species_endpoint}", headers=headers)

            if not response.ok:
                raise Exception("Failed to fetch species from database.")

            response_dict = json.loads(response.text) 
            species_list = response_dict.get("species", [])
            species_of_interest_set = set(species_of_interest_list)

            filtered_species = [
                Species(
                    display_name=specie.get("display_name"),
                    strain=specie.get("strain"),
                    accession=specie.get("accession"),
                    name=specie.get("name"),
                    groups=specie.get("groups", []),
                    taxon_id=specie.get("taxon_id"),
                    release=specie.get("release"),
                    division=specie.get("division"),
                    strain_collection=specie.get("strain_collection"),
                    assembly=specie.get("assembly"),
                    common_name=specie.get("common_name"),
                    aliases=specie.get("aliases", [])
                )
                for specie in species_list if species_of_interest_set.intersection(specie.get("aliases", []))
            ]

            return filtered_species

        except requests.RequestException:
            raise ValueError("Network error while accessing the species database.")
        except json.JSONDecodeError:
            raise ValueError("Error decoding JSON response from the species database.")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")

    def obtain_sequence_from_database(self, DNA_id_list):
        """
        Fetches DNA sequences from the Ensembl database for a list of identifiers.

        Parameters:
        ----------
        DNA_id_list : list of str
            List of DNA identifiers (e.g., Ensembl IDs) to fetch.

        Returns:
        -------
        sequences: list of Sequence objects
            A list of sequence objects containing DNA sequence data.
        """
        try:
            if len(DNA_id_list) == 1:
                headers = {"Content-Type": "application/json"}
                endpoint = f"/sequence/id/{DNA_id_list[0]}"
                response = requests.get(f"{self.server}{endpoint}", headers=headers)
            else:
                headers = {"Content-Type": "application/json", "Accept": "application/json"}
                endpoint = "/sequence/id"
                data = json.dumps({"ids": DNA_id_list})
                response = requests.post(f"{self.server}{endpoint}", headers=headers, data=data)

            if not response.ok:
                raise Exception("Failed to fetch sequences from database.")

            response_data = json.loads(response.text)

            valid_sequences = [
                Sequence(
                    seq_id=data.get("id"),
                    version=data.get("version"),
                    query=data.get("query"),
                    desc=data.get("desc"),
                    molecule=data.get("molecule"),
                    seq=data.get("seq"),
                )
                for data in response_data
            ]

            missing_ids = DNA_id_list
            for sequence in valid_sequences:
                missing_ids.remove(sequence.query)

            return valid_sequences, missing_ids

        except requests.RequestException:
            raise ValueError("Network error while accessing the database.")
        except json.JSONDecodeError:
            raise ValueError("Error decoding JSON response from the database.")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")

    def subsequence_positions(self, sequence_list, subsequence):
        """
        Finds positions of a specific subsequence within a list of DNA sequences.

        Parameters:
        ----------
        sequence_list : list of Sequence
            List of sequence objects to search within.
        subsequence : str
            The subsequence to search for.

        Returns:
        -------
        subsequence_positions: dict
            A dictionary with sequence IDs as keys and lists of positions as values.
        """
        subsequence_positions = {}
    
        for sequence in sequence_list:
            positions = []  
            seq = sequence.seq 
            
            for i in range(len(seq) - len(subsequence) + 1):
                if seq[i:i + len(subsequence)] == subsequence:
                    positions.append(i)  
            
            subsequence_positions[sequence.seq_id] = positions
        
        return subsequence_positions

    def obtain_species_from_sequence(self, sequence_list, species_of_interest):
        """
        Finds positions of a specific subsequence within a list of DNA sequences.

        Parameters:
        ----------
        sequence_list : list of Sequence
            List of sequence objects to search within.
        subsequence : str
            The subsequence to search for.

        Returns:
        -------
        sequences_bool: dict
            A dictionary with sequence IDs as keys and lists of positions as values.
        """
        sequences_bool = {sequence.seq_id: False for sequence in sequence_list}
        assemblies = {species.assembly for species in species_of_interest}
    
        for sequence in sequence_list:
            desc = getattr(sequence, 'desc', None)  
            if desc and any(assembly in desc for assembly in assemblies):
                sequences_bool[sequence.seq_id] = True
        
        return sequences_bool