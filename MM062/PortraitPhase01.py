import numpy as np
import matplotlib.pyplot as plt
import random

def f(Y,t):
    y1, y2 = Y
    return [ -3 * y1 + 2 * y2, y1 - 2 * y2 ]
  
y1 = np.linspace(-10.0, 10.0, 20)
y2 = np.linspace(-10.0, 10.0, 20)

Y1, Y2 = np.meshgrid(y1, y2)

u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = f([x, y],0)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     
Q = plt.quiver(Y1, Y2, u, v, color='r')

plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.savefig('phase-portrait01.png')

from scipy.integrate import odeint

yInit=random.sample(np.arange(-10,10,0.1),50)
xInit=random.sample(np.arange(-10,10,0.1),50)

for i in range(len(xInit)):
    tspan = np.linspace(0, 1, 200)
    y0 = [xInit[i], yInit[i]]
    ys = odeint(f, y0, tspan)
    plt.plot(ys[:,0], ys[:,1], 'b-') # path
    plt.plot([ys[0,0]], [ys[0,1]], 'o') # start
    plt.plot([ys[-1,0]], [ys[-1,1]], 's') # end
    

plt.savefig('phase-portrait-2.png')
plt.show()

