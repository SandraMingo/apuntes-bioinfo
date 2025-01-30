{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true,
    "tags": []
   },
   "source": [
    "You may find useful the Python notes in \n",
    "\n",
    "https://github.com/joseDorronsoro/Notes-on-Python/blob/master/slides_python_short_2020.pdf\n",
    "\n",
    "In particular, chapter 3 deals with strings in Python and chapter 6 deals with files."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Assignment 1: Working with strings\n",
    "\n",
    "\n",
    "### Question 1.\n",
    "\n",
    "The `string` module has a number of useful strings such as `ascii_letters`. Import it into the shell and apply `dir(string)` to find out about them.\n",
    "\n",
    "What does Python consider punctuation marks? And whitespace?\n",
    "\n",
    "What is the function/effect of the various whitespace symbols?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Formatter',\n",
       " 'Template',\n",
       " '_ChainMap',\n",
       " '__all__',\n",
       " '__builtins__',\n",
       " '__cached__',\n",
       " '__doc__',\n",
       " '__file__',\n",
       " '__loader__',\n",
       " '__name__',\n",
       " '__package__',\n",
       " '__spec__',\n",
       " '_re',\n",
       " '_sentinel_dict',\n",
       " '_string',\n",
       " 'ascii_letters',\n",
       " 'ascii_lowercase',\n",
       " 'ascii_uppercase',\n",
       " 'capwords',\n",
       " 'digits',\n",
       " 'hexdigits',\n",
       " 'octdigits',\n",
       " 'printable',\n",
       " 'punctuation',\n",
       " 'whitespace']"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import string\n",
    "\n",
    "dir(string)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n",
      "----------------\n",
      "' \\t\\n\\r\\x0b\\x0c'\n",
      "----------------\n",
      " \t\n",
      "\r",
      "\u000b",
      "\f",
      "\n"
     ]
    }
   ],
   "source": [
    "print(string.punctuation)\n",
    "print(\"----------------\")\n",
    "# why do the next lines show different outputs?\n",
    "print(repr(string.whitespace))\n",
    "print(\"----------------\")\n",
    "print(string.whitespace)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Primero se imprimen todos los símbolos de puntuación que puede contener una cadena de caracters.'repr' muestra los símbolos imprimibles. Un espacio en blanco no se imprime, si no que separa los caracteres, por lo que, cuando se combina con repr, se muestra una representación de caracteres que de otra forma se muestra como un espacio en blanco."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 2.\n",
    "\n",
    "A **protein** can be considered as a sequence of letters from the alphabet list `['A', 'C', 'G', 'T']`.\n",
    "\n",
    "We want to generate random proteins using the method `choice` of the `numpy.random` submodule.\n",
    "\n",
    "Write a function `random_protein(length)` that returns a string with `lenght` characters chosen randomly from `l_bases = ['A', 'C', 'G', 'T']` list.    \n",
    "\n",
    "Hint: apply repeatedly `random.choice` over `l_bases` adding the choices into a list and then convert it into a string using the `join` string method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GCAGTAGTAT\n"
     ]
    }
   ],
   "source": [
    "from numpy.random import choice\n",
    "\n",
    "def random_protein(length):\n",
    "    \"\"\"Generation of a protein with random bases\n",
    "    \"\"\"\n",
    "    bases = ['A', 'C', 'G', 'T']\n",
    "    protein = []\n",
    "    \n",
    "    for i in range(length):\n",
    "        base = choice(bases)\n",
    "        protein.append(base)\n",
    "        \n",
    "    protein_str = ''.join(protein)\n",
    "    \n",
    "    return protein_str\n",
    "\n",
    "print(random_protein(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 3.\n",
    "\n",
    "We are not too sure that our protein string is truly random.\n",
    "\n",
    "To check it write a function `count_bases_0(str_protein)` that returns a dict with keys the bases and values their absolute frequencies (i.e., if at the end we have `'A':153` it means that `'A'` appears 153 times in `str_protein`).   \n",
    "\n",
    "As a suggestion (easy to improve) you can just set a define `d_freqs` where `d_freqs['L']` is a counter for letter `'L'` with an initial value of 0 and simply will traverse the string and increase the value of the corresponding dict's key."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mis freqs {'A': 127, 'C': 126, 'G': 134, 'T': 113} \n",
      "uniform frequency: 125\n"
     ]
    }
   ],
   "source": [
    "def count_bases_0(str_protein):\n",
    "    \"\"\"\n",
    "    \"\"\"\n",
    "    freqs = {'A':0, 'C':0, 'G':0, 'T':0}\n",
    "    \n",
    "    for base in freqs.keys():\n",
    "        freqs[base] = str_protein.count(base)\n",
    "    \n",
    "    return freqs\n",
    "\n",
    "length = 500\n",
    "str_protein = random_protein(length)\n",
    "\n",
    "d_freqs = count_bases_0(str_protein)\n",
    "\n",
    "print(\"mis freqs\", d_freqs, \"\\nuniform frequency: %d\" % (length//4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 4.\n",
    "\n",
    "To work with files in Python we first open them (i.e., get a handle to traverse them) as in `f = open('name.txt', 'r')` and then proceed to read it. Useful methods for this are \n",
    "\n",
    "* `f.read()` which returns a string with the entire file; \n",
    "* `f.readline()` which returns a string with the next line; \n",
    "* `f.readlines()` which returns a list of string with each of the file lines.\n",
    "\n",
    "Write a function `num_lines_chars(f_name)` that returns the number of lines and chars in the text file named `f_name` and compare your results with those of the Linux command `wc -cl f_name`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(447, 7)\n"
     ]
    }
   ],
   "source": [
    "def num_lines_chars(f_name):\n",
    "    \"\"\"We can read the file as a string with the method `.read` for files in section 6 of the pdf\n",
    "    in https://github.com/joseDorronsoro/Notes-on-Python/blob/master/slides_python_short_2020.pdf\n",
    "    \n",
    "    We can also use a counter and increase it each time we use the method '.readline'. \n",
    "    \n",
    "    The number of chars will be the length of the string and since each line ends with '\\n,'\n",
    "    we can count these carriage returns to get the number of lines.\n",
    "    \"\"\"\n",
    "    with open(f_name, \"r\") as f:\n",
    "        content = f.read()    \n",
    "         \n",
    "    \n",
    "    return len(content), content.count(\"\\n\") + 1\n",
    "    \n",
    "f_name = \"lo_que_sea.txt\"\n",
    "print(num_lines_chars(f_name))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
