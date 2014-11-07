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
    tab = np.zeros((len(objets), len(objets)*max(objets.T[1])))
    print tab.shape
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

def pseudo_polynomial_ameliore( objets, epsilone ):
    return 0
    
def schema_approximation():
    return 0

def trier(objets) : 
    rates = np.divide(objets[:,1]*1.0, objets[:,0])
    tpl = [ (rates[i],pv[0],pv[1]) for i,pv in enumerate(objets) ]
    return np.array( [item[1:] for item in sorted(tpl, reverse=True)])
    
def borne_min(objets, pmax) :
    value = 0;
    i = 0;
    while i < len(objets) :
        if objets[i, 0] <= pmax :
            value += objets[i, 1]
            pmax -= objets[i, 0]
        i += 1
    return value

def borne_max(objets, pmax): 
    value = 0;
    i = 0;
    while i<len(objets)-1 and (pmax - objets[i, 0]) >= 0 :
        value += objets[i, 1]
        pmax -= objets[i, 0]
        i += 1
    value += objets[i, 1] * (float(pmax)/objets[i, 0]) 
    return value
    
def recherche(objets, minimum, maximum, pmax):
    #quand pmax = 0 ou qu'il n'y a plus d'objet on est au bout
    if len(objets) == 0 or pmax == 0: 
        return ([], 0); 
    #bornes             
    bmin = borne_min(objets, pmax)
    bmax = borne_max(objets, pmax)
    
    #mise a jour des bornes
    if bmin < minimum :
        minimum = bmin
    
    if bmax < minimum :
        return([], 0)
    
    if pmax - objets[0, 0] < 0 :
        # l'objet est trop lourd on ne l'ajoute pas
        return recherche(objets[1:], minimum, maximum, pmax)
    else :
        # branche ou on ne l'ajoute pas
        s1 = recherche(objets[1:], minimum, maximum, pmax)
        # branche ou on l'ajoute
        s2 = recherche(objets[1:], minimum, maximum, pmax-objets[0, 0])
        if s1[1]>s2[1]+objets[0, 1]:
            return s1
        else :
            # return ([objets[0], s2[0]], s2[1]+objets[0][1])
            try:
                res = np.concatenate((s2[0],objets[0]), axis=1)
            except:
                print s2[0], objets[0]
                raise
            res = np.reshape(res, (2, len(res)>>1))
            return (res, s2[1]+objets[0][1])
            
def algorithme_arborescent(objets, pmax):
    objets = trier(objets)
    maximum = borne_max(objets, pmax)
    minimum = borne_min(objets, pmax)
    solution = recherche(objets, minimum, maximum, pmax);
    print("valeur de la solution : %d"%solution[1])
    return solution[0]
    
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
    # liste_objets = [ tab[j-1, i] - tab[j,i] for j, valeur in enumerate(tab[::-1][:,i]) if j <> 0 ]
    # np.nonzero(np.append( tab[0,i], tab[1:,i] - tab[:-1,i]))
    return 
    # return [ liste_objets[p] for p in np.nonzero(liste_objets) ]

def rand_adp_search(objets, pmax):
    return 0
    
def main():
    pmax = 18
    # objets = []
    # objets = read_file()
    objets = generer_instance()
    # temps_calcul( pseudo_polynomial, objets )
    # tab = pseudo_polynomial(objets)
    # print tab[:,7]
    # trouver_solution( pmax, tab )
    # print trouver_solution( pmax,  pseudo_polynomial(objets) )
    print np.array(algorithme_arborescent(objets, pmax))
    
if __name__ == "__main__":
    main()

