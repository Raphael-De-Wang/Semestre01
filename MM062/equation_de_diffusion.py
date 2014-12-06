# ---- Exercise 2 Equation de diffusion ----
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

tf = 100; nt=100; nx=100; a=1e-3; x=10
dt = tf/float(nt)
dx = x/float(nx) 
x0 = nx/2

U  = np.zeros([nt,nx])
U[0, x0] = 1

for k in range(0, nt-1):
    for i in range(1,nx-1):
        U[k+1,i]= U[k,i]+a*U[k,i+1]-2*U[k,i]+U[k,i-1]

fig = plt.figure()
ax  = fig.gca(projection ='3d')
X   = np.arange(0, x, dx); T = np.arange(0,tf,dt)
X, T= np.meshgrid(X,T)
surf= ax.plot_surface(X, T, U, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth = 0, antialiased =False)
fig.colorbar(surf)
plt.show()
