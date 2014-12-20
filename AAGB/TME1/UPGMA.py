import numpy as np

distTable = np.round(np.random.rand(10,10) * 10)
distTable = distTable + distTable.T
for i,distCol in enumerate(distTable):
    for j,v in enumerate(distCol):
        if i == j:
            distTable[i,j] = np.inf
        
print distTable

class OTU(object):
        def __init__(self, name, postion, distance):
            self.uname = name
            self.dist  = distance
            self.right = None
            self.left  = None

def combineName(iName,jName):
    return "(" + iName + jName + ")"
    
def combineOTU(distTable,OTUs,i,j):
    if i > j:
        i,j = j,i

    new_OTU = OTU(combineName(OTUs[i].uname,OTUs[j].uname), i, distTable[i,j])
    new_OTU.right = OTUs[i]
    new_OTU.left  = OTUs[j]

    OTUs[i] = new_OTU
    OTUs.remove(OTUs[j])
        
def minTable(distTable):
    lig,col = np.shape(distTable)
    pos     = np.argmin(distTable)
    i       = pos / col
    j       = pos % col
    return ( distTable[i,j], i, j )

print minTable(distTable)

def elementCount(otu):
    if otu.left == None and otu.right == None:
        return 1
        
    return elementCount(otu.left) + element(otu.right)

def renewDistTable(distTable,comI,comJ):
    (lig,col) = np.shape(distTable)
    nt = np.array([distTable[i]   for i in range(lig) if i !=comI])
    nt = np.array([distTable[:,i] for i in range(col) if i !=comJ])
    
    
