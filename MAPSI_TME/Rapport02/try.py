import math
import heapq
import numpy as np
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def dessine(x,cname):
    fig = plt.figure()
    plt.imshow(x.reshape(16,16), cmap = cm.Greys_r, interpolation='nearest')
    # plt.show()
    plt.savefig(cname)

dessine(np.arange(16*16),"2")
