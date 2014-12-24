# -*- coding: utf-8 -*-
import copy
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")

def discretisation( X, n_etats ) :
    intervalle = 360. / n_etats
    return np.array([ np.floor( x / intervalle ) for x in X ])

def groupByLabel( y ) :
    index = []
    for i in np.unique( y ): # pour toutes les classes
        ind, = np.where( y == i )
        index.append( ind )
    return index

def learnMarkovModel ( Xc, d ) :
    A = np.zeros((d,d))
    Pi = np.zeros(d)

    for lettre in Xc:
        Pi[lettre[0]] += 1
        for i in range( len(lettre) - 1 ):
            begin = lettre[i]
            to    = lettre[i+1]
            A[begin,to] += 1
        
    A  /= np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi /= Pi.sum()
    return (Pi, A)
    
def stocker_les_modeles (d, X, Y) :
    # paramètre de discrétisation
    Xd = discretisation(X,d) # application de la discrétisation
    index = groupByLabel(Y) # groupement des signaux par classe
    models = []
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        models.append(learnMarkovModel(Xd[index[cl]], d))
    return models

def probaSeq( seq, Pi, A) :
    probs = np.log(Pi[seq[0]])
    for j in range( len(seq) - 1 ) :
        begin = seq[j]
        to    = seq[j+1]
        probs+= np.log(A[begin, to])
    return probs

def probaSeqModele( seq, modeles ) :
    return np.array([ probaSeq(seq, pg[0], pg[1]) for pg in modeles ])
    
def evaluation_des_performances( Xd, Y, models, d ):
    proba = np.array([ [ probaSeq( Xd[i], models[cl][0], models[cl][1]) for i in range( len(Xd) ) ] for cl in range( len( np.unique(Y) ) ) ])
    Ynum = np.zeros(Y.shape)
    for num,char in enumerate(np.unique(Y)):
        Ynum[Y==char] = num
    pred = proba.argmax(0) # max colonne par colonne
    print 'Evaluation Des Performances Résultat [%d]: '%d, np.where(pred != Ynum, 0.,1.).mean()

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

def evaluation_qualitative( X, Y, group, models):
    conf = np.zeros((26,26))
    for i, cls in enumerate(group):
        for echantillon in cls:
            conf[i, np.argmax([ probaSeq( X[echantillon], mdl[0], mdl[1] ) for mdl in models ]) ] += 1
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")

def random_nombre ():
    return np.random.rand()
    
def tirage_selon_loi_sc( SC, rand_num ):
    for i, p in enumerate(SC) :
        if rand_num < p :
            return i

def generate( Pi, A, n ):
    pi = Pi.cumsum()
    a  = np.array([ i.cumsum() for i in A ])
    newa = [ tirage_selon_loi_sc( pi, random_nombre() ) ]
    for i in range( n - 1 ):
        newa += [ tirage_selon_loi_sc( a[newa[i-1]], random_nombre() ) ]
    return np.array(newa)

def iSet_to_Y(itrain,itest,X,Y):
    Xtrain = []
    Ytrain = []
    Xtest  = []
    Ytest  = []

    for i in np.concatenate(itest): 
        Xtest.append(X[i])
        Ytest.append(Y[i])
                
    for i in np.concatenate(itrain):
        Xtrain.append(X[i])
        Ytrain.append(Y[i])

    return np.array(Xtrain),np.array(Ytrain),np.array(Xtest),np.array(Ytest)
            
def main():
    # ---- Classification de lettres manuscrites ----
    data = pkl.load(file("TME6_lettres.pkl","rb"))
    X = np.array(data.get('letters')) # récupération des données sur les lettres
    Y = np.array(data.get('labels')) # récupération des étiquettes associées

    itrain,itest = separeTrainTest(Y,0.8)
    Xtrain,Ytrain,Xtest,Ytest = iSet_to_Y(itrain,itest,X,Y)

    # ---- Biais d'évaluation, notion de sur-apprentissage ----
    for d in xrange(3,30):
        modeles = stocker_les_modeles ( d, Xtrain, Ytrain )
        # modeles = stocker_les_modeles ( d, X, Y )
        evaluation_des_performances( discretisation( Xtest, d ), Ytest, modeles, d )

    # ---- Evaluation qualitative ----
    #evaluation_qualitative( discretisation( X, d ), it_y, itest, modeles)
    '''
    # ---- Modèle génératif ----
    newa = generate(modeles[lettre][0],modeles[lettre][1], 25) # generation d'une séquence d'états
    intervalle = 360./d # pour passer des états => valeur d'angles
    newa_continu = np.array([i*intervalle for i in newa]) # conv int => double
    tracerLettre(newa_continu)
    '''
    
if __name__ == "__main__":
    main()
