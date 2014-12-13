# -*- coding: utf-8 -*-

import math
import time
import numpy as np

def random_list(N):
    return np.round(np.random.rand(N)*100)

liste = random_list(10)
pivot = np.round(np.random.rand()*100)

print "liste: ", liste
print "pivot: ", pivot

def partition(liste, pivot):
    g,d = [],[]
    for el in liste:
        if el > pivot:
            d.append(el)
        elif el < pivot:
            g.append(el)

    return (g,d)

print partition(liste, pivot)

def trouverPivot(liste):
    i = np.around(np.random.rand()*(len(liste)-1))
    return liste[int(i)]

print trouverPivot(liste)

def triRapide(liste):
    if len(liste) <= 1:
        return liste

    pivot = trouverPivot(liste)
    (g,d) = partition(liste,pivot)

    return np.concatenate(([triRapide(g),[pivot],triRapide(d)]))

print "triRapide:\t",triRapide(liste)

def tri(liste):
    if len(liste) <= 1:
        return liste

    pivot = liste[0]
    (g,d) = partition(liste[1:],pivot)

    return np.concatenate(([tri(g),[pivot],tri(d)]))

print "tri:\t", tri(liste)

    
def test(liste):
    t = time.time()
    triRapide(liste)
    t1 = time.time() - t
    print "triRapide Excute Time: ", t1
    
    t = time.time()
    tri(liste)
    t2 = time.time() - t
    print "tri Excute Time: ", t2

def bachmark_test(N):

    liste = random_list(N)
    
    print "1) listes aléatoire"
    test(liste)

    liste.sort()
    
    print "2) listes tirees"
    test(liste)

bachmark_test(100)
