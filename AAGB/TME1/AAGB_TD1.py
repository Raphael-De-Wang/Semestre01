# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy

# 1. UPGMA


class Node:
    def __init__(self, p):
        self.points = p
        self.right = None
        self.left = None
    
def distance(c1, c2):
    dist = .0
    n1 = len(c1.points)
    n2 = len(c2.points)
    for i in xrange(n1):
        for j in xrange(n2):
            p1 = c1.points[i]
            p2 = c2.points[j]
            dim = len(p1)
            d = 0
            for k in xrange(dim):
                d = d + (p1[k]-p2[k])**2
            d = math.sqrt(d)
            dist = dist + d
    dist = dist / (n1*n2)
    return dist

def upgma(points, taxa, k):
    if len(points) == 1:
        return str(taxa ).replace(' ', '').replace('[','(').replace(']',')').replace('\'','')
    nodes = []
    n = len(points)
    for i in xrange(n):
        node = Node([points[i]])
        nodes = nodes + [node]
    nc = n
    while nc > k:
        c1, c2, i1, i2 = 0, 0, 0, 0
        limite = pow(10,9)
        for i in xrange(nc):
            for j in xrange(i+1, nc):
                dis = distance(nodes[i], nodes[j])
                if dis < limite:
                    limite = dis
                    c1 = nodes[i]; c2 = nodes[j];
                    i1 = i; i2 = j;
        node = Node(c1.points + c2.points)
        node.left = c1; node.right = c2;
        
        new_nodes = []
        for i in xrange(nc):
            if i != i1 and i != i2:
                new_nodes = new_nodes + [nodes[i]]
        new_nodes = new_nodes + [node]
        nodes = new_nodes[:]
        nc = nc - 1

    return nodes

def print_cluster(nodes, Matrice_distance, taxa ):
    liste=[]
    for i in xrange(len(nodes)):
        points = nodes[i].points 
        for j in range(len(points)):
            for n, leaf in enumerate(Matrice_distance):
                if points[j] == leaf :
                    liste = np.append(liste, taxa[n])               
        print "cluster", str(i) , liste
        print np.array(points)
    print


# 2. Neighbor joining

def calcDist(distMat, i, j):
   if i < j:
      i, j = j, i
   return distMat[i][j]

def calcDistSum(distMat, i):
   sum = 0

   for k in range(len(distMat)):
      sum += distMat[i][k]

   for k in range(len(distMat)):
      sum += distMat[k][i]

   return sum

def calcQ(distMat, i, j):
   return (len(distMat)-2)*calcDist(distMat, i, j) - calcDistSum(distMat, i) - calcDistSum(distMat, j)

def calcQMat(distMat):
   q = numpy.zeros((len(distMat),len(distMat)), int)

   for i in range(1, len(distMat)):
      for j in range(i):
         q[i][j] = calcQ(distMat, i, j)

   return q

def calcDistOldNew(distMat, i, j):
   return (.5)*(calcDist(distMat, i, j)) + ((1./(2*(len(distMat)-2))) * (calcDistSum(distMat,i) - calcDistSum(distMat, j)))

def calcDistNewOther(distMat, k, f, g):
   return (.5)*(calcDist(distMat,f,k) - calcDistOldNew(distMat, f, g)) + (.5)*(calcDist(distMat,g,k) - calcDistOldNew(distMat, g, f))
   
def minQVal(q):
   iMin = 0
   jMin = 0
   qMin = 0

   for i in range(len(q)):
      for j in range(len(q)):
         if min(qMin, q[i][j]) == q[i][j]:
            qMin = q[i][j]
            iMin = i
            jMin = j

   if i > j:
      i, j = j, i

   return qMin, iMin, jMin

def doNeigJoin(mat, taxaList):
   if len(mat) == 1:
      return str(taxaList).replace(' ', '').replace('[','(').replace(']',')').replace('\'','')

   q = calcQMat(mat)

   minQ, taxaA, taxaB = minQVal(q)
   newMat = numpy.zeros((len(mat)-1, len(mat)-1), float)
   oldTaxaList = taxaList[:]
   oldTaxaList.remove(taxaList[taxaA])
   oldTaxaList.remove(taxaList[taxaB])
   newTaxaList = [[taxaList[taxaA], taxaList[taxaB]]] + oldTaxaList
   for i in range(1, len(newMat)):
      oldI = taxaList.index(newTaxaList[i])
      newMat[i][0] = calcDistNewOther(mat, oldI, taxaB, taxaA)
   for i in range(2, len(newMat)):
      for j in range(1, len(newMat)-1):
         oldI = taxaList.index(newTaxaList[i])
         oldJ = taxaList.index(newTaxaList[j])
         newMat[i][j] = mat[oldI][oldJ]

   return doNeigJoin(newMat, newTaxaList)
   
   
def main():
    #UPGMA
    taxa = [ "A", "B", "C", "D", "E", "F", "G" ]
    Matrice_distance = [ [ 0., 0., 0., 0., 0., 0., 0. ],
             [ 19., 0., 0., 0., 0., 0., 0. ],
             [ 27., 31., 0., 0., 0., 0., 0. ],
             [ 8., 18., 26., 0., 0., 0., 0. ],
             [ 33., 36., 41., 31., 0., 0., 0. ],
             [ 18., 1., 32., 17., 35., 0., 0. ],
             [ 13., 13., 29., 14., 28., 12., 0. ] ]
    print '1. UPGMA Algorithm'
    print taxa
    print (np.array( Matrice_distance))
    print print_cluster(upgma(Matrice_distance, taxa, 3), Matrice_distance, taxa)
    
    #Neighbor Joining
    taxaList = ['A','B','C','D','E','F']
    mat =  [ [ 0., 2., 4., 6., 6., 8.], 
             [ 2., 0., 4., 6., 6., 8. ], 
             [ 4., 4., 0., 6., 6., 8. ], 
             [ 6., 6., 6., 0., 4., 8. ], 
             [ 6., 6., 6., 4., 0., 8. ], 
             [ 8., 8., 8., 8., 8., 0. ]]
    print '2. Neighbor Joining Algorithm'
    print (np.array(taxaList))         
    print (np.array(mat))
    print doNeigJoin(mat, taxaList)
             
main()




