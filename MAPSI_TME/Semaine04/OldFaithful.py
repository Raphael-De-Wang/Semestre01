import numpy as np
import math
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_file ( filename ):
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break
            
    data = []

    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
        
    infile.close ()
    data = np.asarray ( data )
    data.shape = (data.size / 2, 2 )

    return data

def get_params(data):
    mean1 = data[:,0].mean ()
    mean2 = data[:,1].mean ()
    std1 = data[:,0].std ()
    std2 = data[:,1].std ()

    params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                         (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
    
    return params

def normale_bidim ( x, z, params ):
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    a = sigma_x * sigma_z
    b = 1 - (rho**2)
    c = ( ( x - mu_x ) / sigma_x ) ** 2 
    d = (x-mu_x)*(z-mu_z)
    e = ( ( z - mu_z ) / sigma_z ) ** 2
    res = math.exp( ( (2 * rho * d / a) - c - e ) / ( 2 * b ) ) \
              / ( 2 * math.pi * a * math.sqrt(b) ) 
    return res

def dessine_1_normale ( params ):
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)    
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()
    
def dessine_normales ( data, params, weights, bounds, ax ):
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]
    
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)
    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] ) * weights[0]
            norm1 = np.zeros ( (nb_x,nb_z) )
            
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm1[j,i] = normale_bidim ( x[i], z[j], params[1] ) * weights[1]
    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )

def find_bounds ( data, params ):

    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]
    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )
    return ( x_min, x_max, z_min, z_max )

def Q_i( data, params, weights ):

    length = len(data)
    rst = np.zeros((2,length))
    
    for i in range(length):
        rst[0,i] = weights[0] * normale_bidim ( data[i,0], data[i,1], params[0] )
        rst[1,i] = weights[1] * normale_bidim ( data[i,0], data[i,1], params[1] )
    
    smt = rst[0] + rst[1]
    rst[0] = rst[0] / smt
    rst[1] = rst[1] / smt

    return rst
    
def M_step ( data, Q, params, weights ):
    data = data.T
    
    Q0_sum = Q[0].sum()
    Q1_sum = Q[1].sum()
    
    pi_0 = Q0_sum/(Q0_sum+Q1_sum)
    pi_1 = Q1_sum/(Q0_sum+Q1_sum)
    
    mu_x0 = (Q[0]*data[0]).sum()/Q0_sum
    mu_z0 = (Q[0]*data[1]).sum()/Q0_sum
    mu_x1 = (Q[1]*data[0]).sum()/Q1_sum
    mu_z1 = (Q[1]*data[1]).sum()/Q1_sum
    
    sigma_x0 = math.sqrt((Q[0]*((data[0]-mu_x0)**2)).sum()/Q0_sum)
    sigma_z0 = math.sqrt((Q[0]*((data[1]-mu_z0)**2)).sum()/Q0_sum)
    sigma_x1 = math.sqrt((Q[1]*((data[0]-mu_x1)**2)).sum()/Q1_sum)
    sigma_z1 = math.sqrt((Q[1]*((data[1]-mu_z1)**2)).sum()/Q1_sum)
    
    rho0 = (Q[0]*(data[0]-mu_x0)*(data[1]-mu_z0)/(sigma_x0*sigma_z0)).sum()/Q0_sum
    rho1 = (Q[1]*(data[0]-mu_x1)*(data[1]-mu_z1)/(sigma_x1*sigma_z1)).sum()/Q1_sum
    
    n_params = np.array([(mu_x0, mu_z0, sigma_x0, sigma_z0, rho0),
                         (mu_x1, mu_z1, sigma_x1, sigma_z1, rho1)])
    
    n_weights = np.array([pi_0, pi_1])
    
    return (n_params, n_weights)
    
def dessine(data, params, weights):
    bounds = find_bounds ( data, params )
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    dessine_normales ( data, params, weights, bounds, ax )
    plt.show ()
    
def EM (data, params, weights, steps):
    gradients = []
    for i in range(steps):
        dessine(data, params, weights)
        Q = Q_i( data, params, weights )
        [ params, weights ] = M_step ( data, Q, params, weights )
        gradients.append( [ params, weights ] )
    return gradients

def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print "step animate = %d" % ( i )
    
def testcase_mise_au_point(fname):
    data = read_file(fname)
    params = get_params(data)
    weights = np.array ( [ 0.5, 0.5 ] )
    EM (data, params, weights, 20 )

def main():
    fname = "2014_tme4_faithful.txt"
    testcase_mise_au_point(fname)
    '''
    data = read_file(fname)
    params = get_params(data)
    weights = np.array ( [ 0.5, 0.5 ] )
    res_EM = EM (data, params, weights, steps = 20 )
    bounds = find_video_bounds ( data, res_EM )
    fig = plt.figure ()
    ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))
    anim = animation.FuncAnimation(fig, animate, frames = len ( res_EM ), interval=500 )
    # plt.show ()
    anim.save('old_faithful.avi', bitrate=4000)
    '''
if __name__ == "__main__":
    main()

