import numpy as np

def genDistTable(taille,scale=10):
    distTable = np.round(np.random.rand(taille,taille) * scale)
    distTable = distTable + distTable.T
    for i,distCol in enumerate(distTable):
        for j,v in enumerate(distCol):
            if i == j:
                distTable[i,j] = np.inf
    return distTable

def isFeuille(otu):
    if otu.left == None and otu.right == None:
        return True
    return False
    
def countElements(otu):
    if otu.left == None or otu.right == None:
        return 1
    return countElements(otu.right) + countElements(otu.left)
    
class OTU(object):
        def __init__(self, name, haut, right=None, left=None):
            self.uname = name
            self.haut  = haut
            self.right = right
            self.left  = left

        def combineName(self,jName,iName):
            return "( %s , %s )"%(jName,iName)

        def combineOTU(self,distTable,otuList,posI,posJ):
            otu  = otuList[posI]
            haut = distTable[posI,posJ]/2
            self.uname += ":%.2f"%(haut - self.haut)
            otu.uname  += ":%.2f"%(haut - otu.haut)
            newOTU = OTU(self.combineName(self.uname,otu.uname), haut, self, otu)
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

def newDistance(distTable, it, jt, lt, otuList):
    return (distTable[it,lt]*countElements(otuList[it]) + distTable[jt,lt]*countElements(otuList[jt]))/(countElements(otuList[it])+countElements(otuList[jt]))

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
                nt[i,j] = newDistance(distTable, comI, comJ, lt, otuList)
                continue
            nt[i,j] = distTable[it,jt]
    return nt

def upgm(distTable):
    icount = 0
    otuList = newOTUList(len(distTable))
    while len(otuList) > 1:
        print "otuList: ",[ otu.uname for otu in otuList ]
        print "==== Iteration %d ====="%icount
        icount += 1
        comI,comJ = minTable(distTable)
        print comI, comJ, distTable[comI,comJ]
        nt = renewDistTable(distTable,comI,comJ,otuList)
        print nt
        otuList = otuList[comJ].combineOTU(distTable,otuList,comI,comJ)
        distTable = nt
    return otuList
    
taille = 10
distTable = genDistTable(taille)
'''
distTable = np.array([[np.inf,0,0,0,0,0],
                      [2,np.inf,0,0,0,0],
                      [4,4,np.inf,0,0,0],
                      [6,6,6,np.inf,0,0],
                      [6,6,6,4,np.inf,0],
                      [8,8,8,8,8,np.inf]])

distTable = np.array([[np.inf,0,0,0,0],
                      [2,np.inf,0,0,0],
                      [6,6,np.inf,0,0],
                      [4,4,6,np.inf,0],
                      [7,7,9,5,np.inf]])

distTable = distTable + distTable.T

distTable = np.array([[ np.inf, 4., 5., 5., 2. ],
                      [ 4., np.inf, 3., 5., 6. ],
                      [ 5., 3., np.inf, 2., 5. ],
                      [ 5., 5., 2., np.inf, 3. ],
                      [ 2., 6., 5., 3., np.inf ]])

distTable = np.array([[ np.inf,  11.,   8.,   9.,  10.,  10.,   9.,  15.,  10.,   4.],
                      [ 11.,  np.inf,   8.,   8.,   3.,  13.,   8.,   7.,   8.,   6.],
                      [  8.,   8.,  np.inf,  12.,   6.,   8.,   9.,  12.,   5.,  15.],
                      [  9.,   8.,  12.,  np.inf,   9.,   5.,   7.,  17.,  15.,  11.],
                      [ 10.,   3.,   6.,   9.,  np.inf,   5.,  15.,  11.,  14.,   9.],
                      [ 10.,  13.,   8.,   5.,   5.,  np.inf,   4.,  11.,  11.,  10.],
                      [  9.,   8.,   9.,   7.,  15.,   4.,  np.inf,  10.,   9.,  17.],
                      [ 15.,   7.,  12.,  17.,  11.,  11.,  10.,  np.inf,  14.,  15.],
                      [ 10.,   8.,   5.,  15.,  14.,  11.,   9.,  14.,  np.inf,   5.],
                      [  4.,   6.,  15.,  11.,   9.,  10.,  17.,  15.,   5.,  np.inf]])
'''
print distTable

olist = upgm(distTable)
olist[0].show()

