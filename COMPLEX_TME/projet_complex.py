#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time

def read_file( fname="test.in" ):
    with open(fname, 'r') as fb:
        return np.array( [ ( int(line.strip().split()[1]), int(line.strip().split()[2]) ) \
                           for line in fb if len(line.strip().split()) == 3 ] ) 
    
def generer_instance( n=10, low=1, high=10 ): 
    return np.random.randint(low, high, (n,2))

def pseudo_polynomial( objets ):
    tab = np.zeros((len(objets), len(objets) * max(objets.T[1])))
    for i, colonne in enumerate(tab):
        for j, casse in enumerate(colonne):
            if i == 0 :
                if j == 0 :
                    tab[i][j] = 0
                else:
                    tab[i][j] = np.inf
            else :
                tab[i][j] = min(tab[i-1][j], objets[i-1][0]+tab[i-1][j-objets[i-1][1]])
    return tab

def pseudo_polynomial_ameliore( objets ):
    return 0
    
def schema_approximation():
    return 0

def trier(objets):
    rates = np.divide(objets[:,1]*1.0, objets[:,0])
    tpl = [ (rates[i],pv[0],pv[1]) for i,pv in enumerate(objets) ]
    return np.array( [item[1:] for item in sorted(tpl, reverse=True)])

def temps_calcul( lamda, objets ) :
    print "Algorithme %s: "%lamda
    tps1 = time.clock()
    lamda( objets )
    tps2 = time.clock()
    print "Temps d'execution : %d sec"%(tps2-tps1)

def trouver_solution( pmax, tab ) :
    tab = tab[:,::-1]
    for i, valeur in enumerate(tab[-1]) :
        if valeur <= pmax :
            break
    liste_objets = [ tab[j-1, i] - tab[j,i] for j, valeur in enumerate(tab[::-1][:,i]) if j <> 0 ]
    return [ liste_objets[p] for p in np.nonzero(liste_objets) ]

def main():
    # objets =  read_file()
    # temps_calcul( pseudo_polynomial, objets )
    # pmax = 10
    # trouver_solution( pmax,  pseudo_polynomial(objets) )
    # objets = generer_instance()
    objets = read_file()
    print trier(objets)
    
    
if __name__ == "__main__":
    main()
