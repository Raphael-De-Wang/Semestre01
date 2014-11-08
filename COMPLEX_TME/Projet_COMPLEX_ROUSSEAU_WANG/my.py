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

def trier(objets) : 
    rates = np.divide(objets[:,1]*1.0, objets[:,0])
    tpl = [ (rates[i],pv[0],pv[1]) for i,pv in enumerate(objets) ]
    return np.array( [item[1:] for item in sorted(tpl, reverse=True)])
    
def glouton (objets, pmax) :
    
    p_value = 0
    
    for i in range(len(objets)):
        if pmax == 0:
            break
            
        if objets[i, 0] <= pmax :
            p_value += objets[i, 1]
            pmax  -= objets[i, 0]

    return p_value

def borne_max(objets, pmax):
    
    p_value = 0;
    
    for i in range(len(objets)) :
        
        if objets[i, 1] > pmax :
            p_value += objets[i, 1] * (float(pmax)/objets[i, 0])
            break
        
        p_value += objets[i, 1]
        pmax -= objets[i, 0]

    return p_value

def recherche(borne):
    p_glou = 0
    p_bmax = 1
    p_solu = 2
    p_rest = 3
    pmax   = borne[p_rest]
    p_objs = 4
    objets = borne[p_objs]
    p_poid = 0
    p_value = 1

    if pmax < min(objets[:, p_poid]):
        return borne

    sum_p = sum( [ pv[p_poid] for pv in objets] )
    if pmax > sum_p :
        return ( sum_p, sum_p, objets, 0, [] )
    
    if borne[p_bmax] < borne[p_glou] :
        return borne

    if borne[p_bmax] == borne[p_glou] :
        for i in range(len(objets)) :
            if borne[p_rest] > 0 :
                borne[p_solu].append(objets[i])
                borne[p_rest] -= objets[0, p_poid]
                objets = objets[1:]
        return borne
        
    if len(objets) == 0 :
        return borne

    # bornes
    in_pack = sum([ pv[p_value] for pv in borne[p_solu] ])
    borne_l = ( glouton(objets, pmax) + in_pack,     borne_max(objets, pmax) + in_pack,      borne[p_solu].append(objets[0]) , pmax - objets[0,0], objets[1:])
    borne_r = ( glouton(objets[1:], pmax) + in_pack, borne_max(objets[1:], pmax) + in_pack,  borne[p_solu]                   , pmax              , objets[1:])

    while borne_l[p_rest] > 0 and borne_r[p_rest] > 0 :

        if borne_l[p_bmax] < borne_r[p_glou] :
            return recherche(borne_r)
    
        if borne_r[p_bmax] < borne_l[p_glou] :
            return recherche(borne_l)
    
        if borne_l[p_bmax] >= borne_r[p_bmax] :
            borne_l = recherche(borne_l)
        else:
            borne_r = recherche(borne_r)
        
def algorithme_arborescent(objets, pmax):
    objets = trier(objets)
    solution = recherche( (glouton(objets, pmax), borne_max(objets, pmax), [], pmax, objets) )
    print("valeur de la solution : %d"%solution[0])
    return solution[3]
            
def main():
    pmax = 10
    # obj = read_file()
    obj = generer_instance()
    
    print algorithme_arborescent(obj, pmax)
    
if __name__ == "__main__":
    main()
