# -*- coding: utf-8 -*-
import math
import numpy as np

def generer_nouv_tableau(tableau, ligne, col):
    
    nouv_tableau = np.zero((len(tableau)-1, len(tableau)-1))
    
    for (i,j) in tableau:
        d_ik = .5* (Dij(tableau, i, j) - U(tableau, i))
        d_jk = .5* (Dij(tableau, i, j) + U(tableau, i) - U(tableau, j)
        
def tableau_min(Q):
    i, j = 0 
    for t_i, ligne in enumerate(Q):
        t_j = np.argmin(ligne)
        if Q[i,j] > Q[t_i,t_j]:
            i = t_i
            j = t_j
    return i, j

def U(tableau, ligne):
    return sum(tableau[ligne]) / (len(ligne)-2)

def Dij(tableau, ligne, column):
    return tableau[ligne, column]

def initialisation(tableau):
    Q = np.zeros()
    
    for (i,j) in range(len(tableau)):
        if i == j:
            Q[(i,j)] = np.inf
        else:
            Q[(i,j)] = Dij(tableau, i, j) - U(tableau, i) - U(tableau, j)
            
    return Q
    
def neighborjoining(tableau):
    
    for ite in range(len (tableau) ) :
        
        Q = initialisation(tableau)
    
        i,j = tableau_min(Q)
        
        tableau = generer_nouv_tableau(tableau, i, j)
