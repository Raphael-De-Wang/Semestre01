import numpy as np
import matplotlib.pyplot as plt


# ---- Exercise 1 Methode de resoluton numerique dans les systemes dynamiques -----
def f(x):
    return 0.1 * x * ( 30 - x )

def f2(variable):
    [ x, y ] = variable
    return np.array([0.25*x - 0.01*x*y, 0.01*x*y - y])

def init_x(val_init_x, n):
    if type(val_init_x) == type([]):
        x = np.zeros([n,len(val_init_x)])
    else:
        x = np.zeros([n,])
    return x
    
def euler(func, val_init_x, tf = 100, n = 500):
    h = tf/float(n)
    x = init_x(val_init_x, n)
    x[0] = val_init_x
    for i in range(n-1):
        x[i+1] = x[i] + h * func(x[i])
    return x

def RK2(func, val_init_x, tf = 100, n = 500):
    h = tf/float(n)
    x = init_x(val_init_x, n)
    x[0] = val_init_x
    for i in range(n-1):
        x[i+1] = x[i] + (h/2) * func(x[i]) + (h/2) * func(x[i]+h*func(x[i]))
    return x
    
def desine():
    fig = plt.figure()
    points = euler(f,[80,30])
    plt.plot(points[:,0], points[:,1], label='systeme 1, x0=80, y0=30 euler')
    points = RK2(f,[80,30])
    plt.plot(points[:,0], points[:,1], label='systeme 1, x0=80, y0=30 RK2')
    points = euler(f2,[80,30])
    plt.plot(points[:,0], points[:,1],label='systeme 2, x0=80, y0=30 euler')
    points = RK2(f2,[80,30])
    plt.plot(points[:,0], points[:,1],label='systeme 2, x0=80, y0=30 RK2')

    plt.xlabel('Time')
    plt.ylabel('Population Number')
    plt.legend()
    plt.show()

desine()

# ---- Exercise 2 Equation de diffusion ----

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

