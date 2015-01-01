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

x = np.arange(-2*np.pi, 2*np.pi, 0.1)
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(x, np.sin(x), label='Sine')
ax.plot(x, np.cos(x), label='Cosine')
ax.plot(x, np.arctan(x), label='Inverse tan')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, ["abced"], loc='upper left', bbox_to_anchor=(1,1))
ax.grid('on')
fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
