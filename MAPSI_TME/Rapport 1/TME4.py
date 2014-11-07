#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import *
from pylab import *
import matplotlib.pyplot as plt
import math

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = (data.size / 2, 2 )

    return data

data = read_file ( '2014_tme4_faithful.txt' )
print data

def normale_bidim ( x, z, params ):
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    f_x_z= (1/(2*math.pi*sigma_x*sigma_z*math.sqrt(1-rho*rho)))*math.exp((-1/(2*(1-rho*rho)))*(math.pow(((x-mu_x)/sigma_x),2)-2*rho*((x-mu_x)*(z-mu_z)/(sigma_x*sigma_z))+math.pow((z-mu_z)/sigma_z,2)))
    return f_x_z

print normale_bidim ( 3, 5, (0.2,0.5,0.8,1.2,0.5) )

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()
    
    
print dessine_1_normale ( (0.2,0.5,0.8,1.2,0.5) )

def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )


# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.4, 0.6] )
bounds = find_bounds ( data, params )

# affichage de la figure
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()


def Q_i ( data, current_params, current_weights ):
    nb_x = nb_z = 100
   
    bounds = find_bounds ( data, current_params )    
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )    
    tab = [[]]
    a1=np.zeros(100)
    b1=np.zeros(100)
    tab = [[0 for i in xrange(2)] for i in xrange(len(data))]
    current_params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
    current_weights = np.array ( [0.4, 0.6] )
    norm0 = np.zeros ( (nb_x,nb_z) )
    
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], current_params[0] )
            a1[i] += (norm0[j,i])
            a =np.asarray(a1)
    print a
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], current_params[1] )
             b1[i] += (norm1[j,i])
             b = np.asarray(b1)
    print b
    for i in range(len(a)):
        for j in range (len(tab)):
            for k in range(len(tab[j])):
                tab[j][k] = (current_weights[0]*a[i])/((current_weights[0]*a[i])*2) # j=0,1 pour h(m,n)=(alpha_n * N(x_m,mu_n,sigma_n))/(somme de j=1 à N de alpha_j * N(x_m, mu_j, sigma_j))
                tab[j][k] = (current_weights[1]*b[i])/((current_weights[1]*b[i])*2) # j=0,1
    return tab

current_params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
current_weights = np.array ( [0.4, 0.6] )    
Q=Q_i ( data, current_params, current_weights )

def M_step ( data, Q, current_params, current_weights ):
    new_params=[]
    new_weights=[]
    couple=()    
    n = len(data)
    Q_0 = []
    Q_1=[]
    data_0=[]
    data_1=[]
    param_0=[]
    param_1=[]
    for i in range(len(Q)):
        Q_0 = append(Q_0,Q[i][0])
    for i in range(len(data)):
        data_0 = append(data_0,data[:,0])
    for i in range(len(Q)):
        Q_1 = append(Q_1,Q[i][1])
    for i in range(len(data)):
        data_1 = append(data_1,data[:,1])
    for i in range(len(Q_0)):
        for j in range(len(data_0)):
            mu_x_0 = sum(Q_0[i]*data_0[j])/sum(Q_0[i])
            mu_z_0 = sum(Q_0[i]*data_1[j])/sum(Q_0[i])
            pi_0 = sum(Q_0[i]/(sum(Q_0[i])+Q_1[i]))
            sigma_x_0 = sqrt(sum((Q_0[i]*pow((data_1[j]-mu_x_0),2)))/sum(Q_0[i]))
            sigma_z_0 = sqrt(sum(Q_0[i]*pow((data_0[j]-mu_z_0),2))/sum(Q_0[i]))
            rho_0 = sum(Q_0[i]*((data_0[j]-mu_x_0)*(data_1[j]-mu_z_0)/sigma_x_0*sigma_z_0)/sum(Q_0[i])) 
    param_0 = [mu_x_0, mu_z_0, sigma_x_0, sigma_z_0, rho_0] 
    
    for i in range(len(Q_0)):
        for j in range(len(data_0)):
            mu_x_1 = sum(Q_1[i]*data_0[j])/sum(Q_1[i])
            mu_z_1 = sum(Q_1[i]*data_1[j])/sum(Q_1[i])
            pi_1 = sum(Q_1[i]/(sum(Q_0[i])+Q_1[i]))
            sigma_x_1 = np.sqrt(sum(Q_1[i]*pow((data_1[j]-mu_x_1),2))/sum(Q_1[i]))
            sigma_z_1 = np.sqrt(sum(Q_1[i]*pow((data_0[j]-mu_z_1),2))/sum(Q_1[i]))
            rho_1 = sum(Q_1[i]*((data_0[j]-mu_x_1)*(data_1[j]-mu_z_1)/sigma_x_1*sigma_z_1)/sum(Q_1[i])) 
 
    param_1 = [mu_x_1, mu_z_1, sigma_x_1, sigma_z_1, rho_1]

    new_params=np.asarray([param_0,param_1], dtype=float)
    new_weights=[pi_0,pi_1]
    couple=(new_params,new_weights)    
    return couple
    


#initialisation
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )

#algorithme EM
for i in range(4):
    Q_i ( data, params, weights )
    M_step ( data, Q, params, weights )
    dessine_normales ( data, params, weights, bounds, ax )
    i += 1

