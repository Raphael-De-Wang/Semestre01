# -*- coding: utf-8 -*-
import math
import numpy as np

def genDistTable(taille,scale=10):
    distTable = np.round(np.random.rand(taille,taille) * scale)
    distTable = distTable + distTable.T
    for i,distCol in enumerate(distTable):
        for j,v in enumerate(distCol):
            if i == j:
                distTable[i,j] = np.inf
    return distTable

def u(distances):
    sum = 0.
    N   = len(distances)
    for d in distances:
        if d < np.inf:
            sum += d
    return sum

class OTU(object):
        def __init__(self, name, right=None, left=None):
            self.uname = name
            self.right = right
            self.left  = left

        def combineName(self,jName,iName):
            return "( %s , %s )"%(jName,iName)
            
        def combineOTU(self,distTable,otuList,posI,posJ):
            otu  = otuList[posI]
            self.uname += ":%.2f"%((distTable[posI,posJ] + u(distTable[posI]) - u(distTable[posJ]))/2)
            otu.uname  += ":%.2f"%((distTable[posI,posJ] + u(distTable[posJ]) - u(distTable[posI]))/2)
            newOTU = OTU(self.combineName(self.uname,otu.uname), self, otu)
            otuList[posJ] = newOTU
            otuList.remove(otuList[posI])
            return otuList

        def show(self):
            print self.uname

def newOTUList(taille):
    return [ OTU("%d"%i, 0) for i in range(taille) ]

def minTable(distTable):
    lig,col = np.shape(distTable)
    pos     = np.argmin(distTable)
    i       = pos / col
    j       = pos % col
    if i < j:
        i,j = j,i
    return ( i, j )

def net_divergence(distTable):
    nd = np.zeros(np.shape(distTable))
    N = len(distTable)
    for i in range(N):
        for j in range(N):
            if i == j:
                nd[i,j] = np.inf
                continue
            nd[i,j] = distTable[i,j] - (u(distTable[i]) + u(distTable[j]))/(len(distTable)-2)

    return nd

def newDistance(distTable, comI, comJ, lt):
    return ( distTable[comI,lt] + distTable[comJ,lt] - distTable[comI,comJ] ) / 2
    
def renewDistTable(distTable,comI,comJ,otuList):
    it,jt = 0,0
    if comI < comJ:
        comI,comJ = comJ,comI
    (lig,col) = np.shape(distTable)
    nt = np.zeros((lig-1,col-1))
    for i in range(lig-1):
        it = i
        if i > comI - 1:
            it += 1
        for j in range(col-1):
            jt = j
            if j > comI - 1:
                jt += 1
            if i == j:
                nt[i,j] = np.inf
                continue
            if it == comJ or jt == comJ:
                if it == comJ and jt == comJ:
                    continue
                lt = it + jt - comJ
                nt[i,j] = newDistance(distTable, comI, comJ, lt)
                continue
            nt[i,j] = distTable[it,jt]
    return nt

def NJ(distTable):
    icount = 0
    otuList = newOTUList(len(distTable))
    while len(otuList) > 2:
        print "otuList: ",[ otu.uname for otu in otuList ]
        print "==== Iteration %d ====="%icount
        icount += 1
        nd = net_divergence(distTable)
        print nd
        comI,comJ = minTable(nd)
        print comI, comJ, nd[comI,comJ]
        nt = renewDistTable(distTable,comI,comJ,otuList)
        print nt
        otuList = otuList[comJ].combineOTU(distTable,otuList,comI,comJ)
        distTable = nt
    return "( %s: %.2f , %s: %.2f )"%(otuList[0].uname, distTable[0,1]/2, otuList[1].uname, distTable[0,1]/2)

distTable = genDistTable(10)
'''
distTable = np.array([[ 0, 5, 4, 7, 6, 8],
                      [ 0, 0, 7,10, 9,11],
                      [ 0, 0, 0, 7, 6, 8],
                      [ 0, 0, 0, 0, 5, 9],
                      [ 0, 0, 0, 0, 0, 8],
                      [ 0, 0, 0, 0, 0, 0]])
distTable += distTable.T
'''
print distTable
print NJ(distTable)


