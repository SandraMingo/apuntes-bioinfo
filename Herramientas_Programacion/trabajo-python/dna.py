import random
import os

class DNA:

    def __init__(self):
        self.dna_chain = ''
        self.frequencies = {'A':0, 'T':0, 'C':0, 'G':0}
        self.nucleotides = ['A', 'T', 'G', 'C']
        self.complementary = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        self.saved = False
        self.directory = None

    def create_dna_chain(self, length):
        ''' Creation of a DNA chain
        Input Parameters
        ----------------
        length: int
            number of nucleotides of the new chain
        Return
        ------
        self.dna_chain: str
            string with the nucleotide sequence of the created chain
        '''
        dna_sequence = random.choices(self.nucleotides, k = length)
        self.dna_chain = ''.join(dna_sequence)
        self.saved = False
        return self.dna_chain

    def validate_chain(self):
        ''' Validates if the current DNA chain has only nucleotides or something else.
        Return
        ------
        validation: bool
            True when the chain contains only the nucleotides A, T, C, G
            False when it includes something else
        '''
        validation = True
        for nucleotide in self.dna_chain:
            if nucleotide not in self.nucleotides:
                validation = False
                break
        return validation

    def mutate_chain(self, number_changes):
        '''Changes in the current DNA sequence a number of random positions into a different, random nucleotide
        Input Parameter
        --------------
        number_changes: int
            Number of positions that should be mutated/modified
        Return
        ------
        None
            number of changes exceeds the number of nucleotides in the chain
        self.dna_chain: str
            string with the new nucleotide sequence including the mutations
        '''
        #First validate that the number of changes can be done in the chain
        if number_changes > len(self.dna_chain):
            return

        #Convert DNA chain in a list for item assignment
        sequence_list = list(self.dna_chain)
        #Get the random positions to be mutated
        random_positions = random.sample(range(len(self.dna_chain)), number_changes)
        for position in random_positions:
            current_nucleotide = sequence_list[position]
            #Remove original nucleotide from the possible nucleotides for the mutation
            possible_nucleotides = self.nucleotides[:]
            possible_nucleotides.remove(current_nucleotide)
            #Mutate the nucleotide to a new random one
            sequence_list[position] = random.choice(possible_nucleotides)
        self.dna_chain = ''.join(sequence_list)
        self.saved = False
        return self.dna_chain

    def measure_frequencies(self):
        ''' Measure the frequencies of each nucleotide in the current DNA chain
        Return
        ------
        null:
            The chain is not valid
        self.frequencies: dict
            Dictionary with each nucleotide as key and the number of appearance (its frequency) as value
        '''
        self.frequencies = {
            'A': self.dna_chain.count('A'),
            'T': self.dna_chain.count('T'),
            'C': self.dna_chain.count('C'),
            'G': self.dna_chain.count('G')
        }
        return self.frequencies

    def count_subsequence(self, subsequence):
        '''Counts the times a given subsequence appears in the current DNA sequence
        Input Parameter
        ---------------
        subsequence: str
            A subsequence to look for in the DNA sequence
        Return
        ------
        counter: int
            The number of times the subsequence was found in the current DNA sequence
        '''
        #count does not repeat positions, so I'm using a for loop to simulate a sliding window
        counter = 0
        for i in range(len(self.dna_chain)):
            if self.dna_chain[i:i+len(subsequence)] == subsequence:
                counter += 1
        return counter

    def synthesize_complementary_reverse(self):
        '''Obtain the complementary and reverse strands of the current DNA sequence
        Returns
        -------
        complementary_strand: str
            Complemented DNA sequence of the current one, changing A <-> T, and C <-> G
        reverse_strand: str
            Current DNA sequence in reverse order
        '''
        reverse_strand = self.dna_chain[::-1]

        complementary_strand = ''.join(self.complementary[nucleotide] for nucleotide in self.dna_chain)

        return complementary_strand, reverse_strand

    def measure_gc_content(self):
        '''Measures the percentage of the nucleotides GC in the current DNA sequence
        Return
        ------
        gc_content / len(self.dna_chain) * 100 : float
            GC content expressed in %
        '''
        frequencies = self.measure_frequencies()
        gc_content = frequencies['G'] + frequencies['C']
        return gc_content / len(self.dna_chain) * 100
