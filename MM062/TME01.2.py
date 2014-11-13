import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

tf = 100
nt = 100
nx = 100
a = 0.001
x = 10
dt = x / float(nt)
dx = x / float(nx)
n0 = nx / 2.

U= np.zeros([nt,nx])
U[0,x0]=1

for k in range (0,nt-10):
    for i in range (1,nx-1):
        U[k+1,i]=

