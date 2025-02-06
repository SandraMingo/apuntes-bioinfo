def random_protein(length):
    """Generation of a protein with random bases
    """
    bases = ['A', 'C', 'G', 'T']
    protein = []
    
    for i in range(length):
        base = choice(bases)
        protein.append(base)
        
    protein_str = ''.join(protein)
    
    return protein_str
def count_bases_0(str_protein):
    """
    """
    freqs = {'A':0, 'C':0, 'G':0, 'T':0}
    
    for base in freqs.keys():
        freqs[base] = str_protein.count(base)
    
    return freqs
def num_lines_chars(f_name):
    """We can read the file as a string with the method `.read` for files in section 6 of the pdf
    in https://github.com/joseDorronsoro/Notes-on-Python/blob/master/slides_python_short_2020.pdf
    
    We can also use a counter and increase it each time we use the method '.readline'. 
    
    The number of chars will be the length of the string and since each line ends with '\n,'
    we can count these carriage returns to get the number of lines.
    """
    with open(f_name, "r") as f:
        content = f.read()    
         
    
    return len(content), content.count("\n") + 1
    
