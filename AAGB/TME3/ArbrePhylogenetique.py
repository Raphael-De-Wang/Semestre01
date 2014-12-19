# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#  ==== Variables globales : ====

# Les transcrits pour le gene MAPK8 :
Human       = ["abcd", "acde", "abcdeh","abcdefiIjnpq","abcdefiIjnpz", "abcdegiIjnpz","abcdegiIjnpq","abcdeIjnpz"]
Mouse       = ["qpnjIigedcba","qpnjIifedcba","zpnjIigedcba","zpnjIedcba","zpnjIifedcba"]
Xenopus     = ["rpljIigedcba","qpomjIigedcba"]
ZebrafishA  = ["qpnjIifedcba","qpnjIigedcba"]
ZebbrafishB = ["zpnjIigedcba", "qpnjIigedcba", "gedcba"]
FuguA       = ["abcdefijk","abcdeIjnpz","abcdegiIjnpq","abcdefiIjnpq"]
FuguB       = ["abcdeIjnpz","abcdegiIjnpq","abcdefiIjnpq"]
Drosophila  = ["pnjiIgedcba"]
Celegans    = ["stuabcdegiIjn", "uabcdegiIjn"]

# C est une matrice 3x3 qui permet de fixer un cout de changement d’état des exons.
C = 1 - np.eye(3,3)

# E est l’ensemble des exons observé pour MAPK8 :
E = ["a","b","c","d","e","f","g","h","i","I","j","k","l","m","n","o","p","q","z","r","s","t","u"]

def MAPK8():
    # Construire l'arbre
    G = nx.DiGraph()
    
    # Feuilles
    
    G.add_node( 2, nName = 'Ce', costTab = None, trans = Celegans )
    G.add_node( 4, nName = 'Dr', costTab = None, trans = Drosophila )
    G.add_node(12, nName = 'Za', costTab = None, trans = ZebrafishA )
    G.add_node(13, nName = 'Fa', costTab = None, trans = FuguA )
    G.add_node(14, nName = 'Zb', costTab = None, trans = ZebbrafishB )
    G.add_node(15, nName = 'Fb', costTab = None, trans = FuguB )
    G.add_node(16, nName = 'Hu', costTab = None, trans = Human )
    G.add_node(17, nName = 'Mo', costTab = None, trans = Mouse )
    G.add_node(11, nName = 'Xe', costTab = None, trans = Xenopus )

    G.add_node( 1, nName = 'N1', costTab = None)
    G.add_node( 3, nName = 'N3', costTab = None)
    G.add_node( 5, nName = 'N5', costTab = None)
    G.add_node( 6, nName = 'N6', costTab = None)
    G.add_node( 7, nName = 'N7', costTab = None)
    G.add_node( 8, nName = 'N8', costTab = None)
    G.add_node( 9, nName = 'N9', costTab = None)
    G.add_node(10, nName = 'N10', costTab = None)
    
    # edges
    G.add_edge(8,12)
    G.add_edge(8,13)
    G.add_edge(9,14)
    G.add_edge(9,15)
    G.add_edge(10,16)
    G.add_edge(10,17)
    G.add_edge(6,8)
    G.add_edge(6,9)
    G.add_edge(7,10)
    G.add_edge(7,11)
    G.add_edge(5,6)
    G.add_edge(5,7)
    G.add_edge(3,4)
    G.add_edge(3,5)
    G.add_edge(1,2)
    G.add_edge(1,3)

    return G
    
G = MAPK8()
racine = 1
# nx.draw(G)
# plt.show()
# print G.node[racine]

# 1. Ecrire la fonction exonState qui prend une liste de transcrits et renvoie exS l’état de chaque exon de E dans cette liste.

def exonCount(tran, E):
    tranj = "".join(tran)
    return np.array([ tranj.count(e) for i, e in enumerate(E) ])

# le vecteur exS de longueur E à valeurs dans 0,1,2 codant pour { absent, alternativement présent, constitutivement présent}
def exonCodant(tran, count):
    code = []
    for c in count:
        if c == 0:
            code.append(0)
        elif c > 0 and c < len(tran):
            code.append(1)
        elif c == len(tran):
            code.append(2)
        else:
            raise Exception("InvaildExonCountValue")
            
    return np.array(code)
    
def exonState(trans, E):
    return np.array([ exonCodant(t, exonCount(t, E)) for t in trans ])

# print exonState([Human,Mouse,Xenopus,ZebrafishA,ZebbrafishB,FuguA,FuguB,Drosophila,Celegans], E)

def tranCostTab(costTab, C):
    return np.array([ [ min(cost + c) for c in C] for cost in costTab])

def isFeuille(A,racine):
    return len( A.successors(racine) ) == 0
    
def sankUp(A,C,E,racine):
    if (isFeuille(A,racine)):
        ec = exonCodant(A.node[racine]['trans'], exonCount(A.node[racine]['trans'], E))
        A.node[racine]['exonState'] = np.array([ [e] for e in ec])
        A.node[racine]['costTab']   = np.ones((len(E),3)) * np.inf
        for i,e in enumerate(ec):
            A.node[racine]['costTab'][i,e] = 0;
            
    else:
        [ cl, cr ] = A.successors(racine)
        lcostTab  = sankUp(A,C,E,cl)['costTab']
        rcostTab  = sankUp(A,C,E,cr)['costTab']
        A.node[racine]['costTab'] = tranCostTab( lcostTab, C) + tranCostTab( rcostTab, C)
        A.node[racine]['exonState']   = np.array([ [ i for i,c in enumerate(cost) if min(cost) == c ] for cost in A.node[racine]['costTab']])
        
    return A.node[racine]

sankUp(G,C,E,racine)

print G.node[racine]



