#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time

def read_file( fname="test.in" ):
    with open(fname, 'r') as fb:
        return np.array( [ ( int(line.strip().split()[1]), int(line.strip().split()[2]) ) \
                           for line in fb if len(line.strip().split()) == 3 ] ) 
    
def generer_instance( n=10, low=1, high=100 ): 
    return np.random.randint(low, high, (n,2))

"""
NE MARCHE PAS UTILISER trouver_solution2
"""

def trouver_solution(pmax, tab, objets) :
    tab = tab[::-1,::-1]
    for i, valeur in enumerate(tab[0]) :
        if valeur <= pmax :
            vmax = i
            break
            
    valeur = tab[0, vmax]
    liste_objets = []
    for i, v in enumerate(tab[:,vmax]):
        if v <> valeur :
            liste_objets.append(objets[len(tab)-i])
            valeur = v
    return liste_objets
    
def trouver_solution2(pmax, tab, objets, k=1) :
    ivmax = 0
    ligne = tab[-1] 
    liste_objets = []   
    for i in range(len(ligne)-1, 0, -1) :
        if ligne[i] <= pmax :
            ivmax = i
            break  
    valeur = tab[-1, ivmax]
    for i in range(len(tab)-1, -1, -1) :
        if tab[i, ivmax] <> valeur :
            liste_objets.append(objets[i])
            ivmax = ivmax - int(round((objets[i][1]/k)))
            valeur = tab[i, ivmax]
    print "valeur de la solution optimale : %d"%sum(np.array(liste_objets)[:, 1])
    return liste_objets
    
    
    
def generer_tab(objets) :
    vmax = sum(objets.T[1])+1
    tab = np.zeros((len(objets)+1, vmax))
    for i in range(len(objets)+1) : 
        # for j in range(len(objets)*vmax):
        for j in range(vmax):
            if i == 0 :
                if j == 0 :
                    tab[i][j] = 0
                else:
                    tab[i][j] = np.inf
            else :
                tab[i][j] = min(tab[i-1][j], objets[i-1][0]+tab[i-1][j-objets[i-1][1]])

    return tab
    
def pseudo_polynomial( objets , pmax):
    tab = generer_tab( objets )
    return trouver_solution2(pmax, tab, objets)

def generer_tab_approxime( obj, epsilone ) :

    objets = np.empty_like (obj)
    np.copyto(objets, obj)
    vmax = sum(objets.T[1])+1
    k = ((epsilone * vmax / len(objets)))
    vmax = int(round(vmax / k))
    for i in range(len(objets)) :
        objets[i][1] = int(round(objets[i][1] / k))
    tab = np.zeros((len(objets)+1, vmax+1))
    for i in range(len(objets)+1) : 
        for j in range(vmax+1):
            if i == 0 :
                if j == 0 :
                    tab[i][j] = 0
                else:
                    tab[i][j] = np.inf
            else :
                tab[i][j] = min(tab[i-1][j], objets[i-1][0]+tab[i-1][j-objets[i-1][1]])
    return tab, k

def pseudo_polynomial_approxime( objets , pmax, epsilone):
    tab, k = generer_tab_approxime( objets, epsilone )
    return trouver_solution2(pmax, tab, objets, k)


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
    
def recherche(objets, minimum, pmax):
    #quand pmax = 0 ou qu'il n'y a plus d'objet on est au bout
    if len(objets) == 0 or pmax == 0: 
        return ([], 0); 
    #bornes             
    bmin = borne_min(objets, pmax)
    bmax = borne_max(objets, pmax)
    
    #mise a jour des bornes
    if bmin > minimum :#modification
        minimum = bmin
    
    if bmax < minimum :
        return([], 0)
    
    if pmax - objets[0, 0] < 0 :
        # l'objet est trop lourd on ne l'ajoute pas
        return recherche(objets[1:], minimum, pmax)
    else :
        # branche ou on ne l'ajoute pas
        s1 = recherche(objets[1:], minimum, pmax)
        if s1[1] > minimum :
            minimum = s1[1]
        # branche ou on l'ajoute
        s2 = recherche(objets[1:], minimum-objets[0, 1], pmax-objets[0, 0])#modification de minimum
        if s1[1]>s2[1]+objets[0, 1]:
            return s1
        else :
            # return ([objets[0], s2[0]], s2[1]+objets[0][1])
            res = [s2[0],objets[0]]
            return (res, s2[1]+objets[0][1])

def algorithme_arborescent(objets, pmax):
    objets = trier(objets)
    minimum = borne_min(objets, pmax)
    solution = recherche(objets, minimum, pmax)
    print("valeur de la solution : %d"%solution[1])
    return solution[0]
    
            
def parcourtAleatoire(objets, pmax):
    if len(objets) == 0 or pmax == 0 :
        return ([], 0)
    else :
        if pmax - len(objets) < 0 : 
            return parcourtAleatoire(objets[1:], pmax)
        else:
            if np.random.rand()  < 0.5 :
                s = parcourtAleatoire(objets[1:], pmax-objets[0, 0])
                return ((objets[0], s[0]), s[1]+objets[0][1])
            else : 
                return parcourtAleatoire(objets[1:], pmax)
    
def rasp(temps, objets, pmax) :
    t1 = time.clock()
    sopt = ([],0)
    while(time.clock() - t1 < temps) :
        s2 = parcourtAleatoire(objets, pmax)
        if s2[1] > sopt[1]:
            sopt = s2
    print("Valeur de la solution optimale : %d"%sopt[1])
    return sopt[0]

def temps_calcul( lamda, objets, pmax ) :
    print "Algorithme %s: "%lamda
    tps1 = time.clock()
    lamda( objets, pmax )
    tps2 = time.clock()
    print "Temps d'execution : %d sec"%(tps2-tps1)

def main():
    pmax = 3022
    temps = 5
    epsilone = 0.5
    obj = read_file()

    #obj = generer_instance(1000)
    #obj = np.array([[1, 3],[3, 1],[6, 3], [6, 7], [6, 2]])
    
    objets = np.empty_like (obj)   
    #print obj
    np.copyto(objets, obj)
    print "Algorithme pseudo_polynomial : "
    print pseudo_polynomial( objets , pmax)       
    '''np.copyto(objets, obj)
    print "\n\nAlgorithme pseudo-polynomial approché : "
    print pseudo_polynomial_approxime( objets , pmax, epsilone)    
    '''
    np.copyto(objets, obj)
    print "\n\nAlgorithme arborescent : "
    print algorithme_arborescent(objets, pmax)
    '''
    print "\n\nAlgorithme Randomized Adaptative Search Procedure:"
    np.copyto(objets, obj)
    print rasp(temps, objets, pmax)
    '''
    
if __name__ == "__main__":
    main()
