# %% [markdown]
# You may find useful the Python notes in 
# 
# https://github.com/joseDorronsoro/Notes-on-Python/blob/master/slides_python_short_2020.pdf
# 
# In particular, chapter 3 deals with strings in Python and chapter 6 deals with files.

# %% [markdown]
# ## Assignment 1: Working with strings
# 
# 
# ### Question 1.
# 
# The `string` module has a number of useful strings such as `ascii_letters`. Import it into the shell and apply `dir(string)` to find out about them.
# 
# What does Python consider punctuation marks? And whitespace?
# 
# What is the function/effect of the various whitespace symbols?

# %%
import string

dir(string)


# %%
print(string.punctuation)
print("----------------")
# why do the next lines show different outputs?
print(repr(string.whitespace))
print("----------------")
print(string.whitespace)

# %% [markdown]
# Primero se imprimen todos los símbolos de puntuación que puede contener una cadena de caracters.'repr' muestra los símbolos imprimibles. Un espacio en blanco no se imprime, si no que separa los caracteres, por lo que, cuando se combina con repr, se muestra una representación de caracteres que de otra forma se muestra como un espacio en blanco.

# %% [markdown]
# ### Question 2.
# 
# A **protein** can be considered as a sequence of letters from the alphabet list `['A', 'C', 'G', 'T']`.
# 
# We want to generate random proteins using the method `choice` of the `numpy.random` submodule.
# 
# Write a function `random_protein(length)` that returns a string with `lenght` characters chosen randomly from `l_bases = ['A', 'C', 'G', 'T']` list.    
# 
# Hint: apply repeatedly `random.choice` over `l_bases` adding the choices into a list and then convert it into a string using the `join` string method.

# %%
from numpy.random import choice

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

print(random_protein(10))

# %% [markdown]
# ### Question 3.
# 
# We are not too sure that our protein string is truly random.
# 
# To check it write a function `count_bases_0(str_protein)` that returns a dict with keys the bases and values their absolute frequencies (i.e., if at the end we have `'A':153` it means that `'A'` appears 153 times in `str_protein`).   
# 
# As a suggestion (easy to improve) you can just set a define `d_freqs` where `d_freqs['L']` is a counter for letter `'L'` with an initial value of 0 and simply will traverse the string and increase the value of the corresponding dict's key.

# %%
def count_bases_0(str_protein):
    """
    """
    freqs = {'A':0, 'C':0, 'G':0, 'T':0}
    
    for base in freqs.keys():
        freqs[base] = str_protein.count(base)
    
    return freqs

length = 500
str_protein = random_protein(length)

d_freqs = count_bases_0(str_protein)

print("mis freqs", d_freqs, "\nuniform frequency: %d" % (length//4))

# %% [markdown]
# ### Question 4.
# 
# To work with files in Python we first open them (i.e., get a handle to traverse them) as in `f = open('name.txt', 'r')` and then proceed to read it. Useful methods for this are 
# 
# * `f.read()` which returns a string with the entire file; 
# * `f.readline()` which returns a string with the next line; 
# * `f.readlines()` which returns a list of string with each of the file lines.
# 
# Write a function `num_lines_chars(f_name)` that returns the number of lines and chars in the text file named `f_name` and compare your results with those of the Linux command `wc -cl f_name`.

# %%
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
    
f_name = "lo_que_sea.txt"
print(num_lines_chars(f_name))


