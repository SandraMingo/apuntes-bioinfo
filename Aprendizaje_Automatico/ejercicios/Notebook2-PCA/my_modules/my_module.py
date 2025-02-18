#Example of definition of a new module.
#A module is a file containing Python definitions and statements.
#The file name is the module name with the suffix .py appended.
#Within a module, the module’s name (as a string) is available as
#the value of the global variable __name__.
import numpy as np
import pandas as pd
import os
import struct
import matplotlib.pyplot as plt


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def read_MNIST():
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    fname_img = './data/t10k-images.idx3-ubyte'
    fname_lbl = './data/t10k-labels.idx1-ubyte'
                 
   
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


#You can define new functions

##def my_function(variables):
##    Your code
##    return variables
