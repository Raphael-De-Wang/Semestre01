# -*- coding: utf-8 -*-
import time
import numpy as np
import pickle as pkl
import os.path as op

# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
def load_data(fname):
    data = pkl.load(file( fname, "rb"))
    X = np.array(data.get('letters'))
    Y = np.array(data.get('labels'))
    nCl = 26
    return data, X, Y, nCl
    
#### Apprentissage d'un modèle connaissant les états ####
def discretisation( X, n_etats = 10 ) :
    intervalle = 360. / n_etats
    return np.array([ np.floor( x / intervalle ) for x in X ])

# Hypothèse Gauche-Droite
def seq_Xd(Xd,N):
    return np.floor(np.linspace(0,N-.00000001,len(Xd)))
    
def initGD(X,N):
    return np.array([ seq_Xd(x,N) for x in X ])

# Apprentissage
def learnHMM(allx, allq, N, K, initTo0=True):
    
    if np.shape(allx) <> np.shape(allq):
        raise Exception("Invalid Data")
        
    if initTo0:
        A  = np.zeros((N,N))
        B  = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A  = np.ones((N,N))*eps
        B  = np.ones((N,K))*eps
        Pi = np.ones(N)*eps

    for i in range(len(allx)):
        lettre   = allx[i]
        etat_seq = allq[i]
        Pi[etat_seq[0]] += 1
        
        for j in range( len(lettre) - 1 ):
            begin = etat_seq[j]
            to    = etat_seq[j+1]
            A[begin,to] += 1
            
        for j in range( len(lettre) ):
            n = etat_seq[j]
            k = lettre[j]
            B[n,k] += 1
            
    A /= np.maximum(A.sum(1).reshape(N,1),1)
    B /= np.maximum(B.sum(1).reshape(N,1),1)
    Pi/= Pi.sum()

    return ( Pi, A, B )

# Viterbi (en log)
def __psi(delta_t_1, A, N):
    return [ np.argmax(np.log(A[:,j]) + delta_t_1) for j in range(N) ]

def __delta(delta_t_1, x, A, B, N):
    return np.array([np.max(np.log(A[:,j]) + delta_t_1) + np.log(B[j,x[-1]]) for j in range(N)])
    
def delta(x,Pi,A,B,N):
    if len(x) == 1:
        return np.log(Pi) + np.log(B[:,x[-1]]), [np.array([-1 for i in range(len(Pi))])]
        
    dt_1, psi_table = delta(x[:-1],Pi,A,B,N)
    
    psi_table.append(__psi(dt_1,A,N))
    
    return __delta(dt_1, x, A, B, N), psi_table

def viterbi(x,Pi,A,B):
    # 2. Récursion:
    delta_t, psi_table = delta(x,Pi,A,B,len(Pi))
    psi_table = np.array(psi_table)
    delta_t   = np.array(delta_t)

    # 3. Terminasion:
    p_est = max(delta_t)

    # 4. Chemin
    T = len(x)
    s_est = np.zeros(T)
    s_est[T-1] = np.argmax(delta_t)

    for t in range(T-2, -1, -1):
        s_est[t] = psi_table[t+1, s_est[t+1]]

    return ( p_est, s_est )

def methode_alpha(x, Pi, A, B):
    N = len(Pi)
    if len(x) == 1:
        return Pi * B[:,x[-1]]
        
    alpha_t_1 = methode_alpha(x[:-1], Pi, A, B)
    return [ sum( alpha_t_1 * A[:,j] ) + B[j,x[-1]] for j in range(N) ]

# [OPT] Probabilité d'une séquence d'observation
def calc_log_probs_v2(x, Pi, A, B):
    return np.log(sum(methode_alpha(x, Pi, A, B)))

# print calc_log_probs_v2(Xd[0], Pi, A, B)

#### Apprentissage complet (Baum-Welch simplifié) ####

# apprendre les modèles correspondant aux 26 lettres de l'alphabet.
def baum_welch_simplifie( lv_lst, X, Y, N = 5, K = 10):
    # 1. Initialisation des états cachés arbitraire (eg méthode gauche-droite)
    nom_iter = 0
    alpha    = []
    Xd       = discretisation(X,K)
    q        = initGD(X,N)
    
    # 2. Tant que critère de convergence non atteint
    while True:
        if ( nom_iter > 2 ) and ( lv_lst[-1] - lv_lst[-2] < 0.0001 ):
            break
            
        lv        = 0
        nom_iter += 1
        alpha     = []
        for lettre in alphabet:
            # 1. Apprentissage des modèles
            ( Pi, A, B ) = learnHMM( Xd[Y==lettre], q[Y==lettre], N, K)
            # print "nom_iter: %d"%nom_iter, B
            alpha.append((Pi,A,B))
            # 2. Estimation des états cachés par Viterbi
            for [i] in np.argwhere(Y==lettre):
                ( p_est, s_est ) = viterbi(Xd[i],Pi,A,B)
                q[i] = s_est
                lv  += p_est
                
        lv_lst.append(lv)
                
    return alpha

# Evaluation des performances
def groupByLabel( y ) :
    index = []
    for i in np.unique( y ): # pour toutes les classes
        ind, = np.where( y == i )
        index.append( ind )
    return index

# Récupérer la méthode separeTrainTest(y, pc) documentée lors de la séance précédente.
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

# Evaluer les performances sur les données de test.
def evaluation_des_performances( Xd, Y, models, d ):
    proba = np.array([ [ viterbi( Xd[i], models[cl][0], models[cl][1], models[cl][2])[0] for i in range( len(Xd) ) ] for cl in range( len( np.unique(Y) ) ) ])
    Ynum = np.zeros(Y.shape)
    for num,char in enumerate(np.unique(Y)):
        Ynum[Y==char] = num
    pred = proba.argmax(0) # max colonne par colonne
    resultat = np.where(pred != Ynum, 0.,1.).mean()
    var      = np.where(pred != Ynum, 0.,1.).var()
    print 'Evaluation Des Performances Résultat [%d]: '%d, resultat
    return resultat,var

#### Génération de lettres ####
# Comme dans le TME précédent, proposer une procédure de génération de lettres.
# Donner le code de generateHMM qui génère une séquence d'observations (et une séquence d'états) à partir des sommes cumulées des modèles de lettres (cf usage ci-dessous)

def random_prendre(distProbs):
    alea = np.random.rand()
    for i in xrange(len(distProbs)):
        if alea < distProbs[i]:
            return i

def generateHMM(Pic, Ac, Bc, longeur):
    s = [ random_prendre(Pic) ]
    x = [ random_prendre( Bc[s[0]]) ]
    for i in xrange(longeur - 1):
        s.append(random_prendre(Ac[s[i]]))
        x.append(random_prendre(Bc[s[i+1]]))

    return s,x

#### Rapport 02 ####

def iSet_to_Y(ia,X,Y):
    Xt = []
    Yt = []

    for i in np.concatenate(ia): 
        Xt.append(X[i])
        Yt.append(Y[i])
                
    return np.array(Xt),np.array(Yt)

def iat (itrain, itest, Y_indice) :
    ia_x = []
    ia_y = []
    for i, cls in enumerate( itrain ) :
        ia_x += cls.tolist()
        ia_y += [ Y_indice[i] for j in range( len(cls) ) ]

    ia_x = np.array(ia_x)
    ia_y = np.array(ia_y)
        
    it_x = []
    it_y = []
    for i, cls in enumerate( itest ):
        it_x += cls.tolist()
        it_y += [ Y_indice[i] for j in range( len(cls) ) ]

    it_x = np.array(it_x)
    it_y = np.array(it_y)

    return ( ia_x, ia_y, it_x, it_y )

def main(data, X, Y, nCl):
    data = pkl.load(file("TME6_lettres.pkl","rb"))
    X = np.array(data.get('letters')) # récupération des données sur les lettres
    Y = np.array(data.get('labels')) # récupération des étiquettes associées

    itrain,itest = separeTrainTest(Y,0.8)
    Xtrain,Ytrain = iSet_to_Y(itrain,X,Y)
    Xtest,Ytest   = iSet_to_Y(itest,X,Y)
    Y_indice = np.unique(Y)
    ( ia_x, ia_y, it_x, it_y ) = iat (itrain, itest, Y_indice)
    
    # ---- Biais d'évaluation, notion de sur-apprentissage ----
    for N in range(2,8):
        print "N = %d"%N
        if op.exists("Res[N=%d,K=(5,200)].npy"%(N)):
            continue
        trainRes = []
        testRes  = []
        trainVarRes = []
        testVarRes  = []
        kRange = range(5,200)
        for K in kRange:
            modeles = baum_welch_simplifie( [], Xtrain, Ytrain, N, K)
            aMean, aVar = evaluation_des_performances( discretisation( Xtrain, K ), Ytrain, modeles, K )
            tMean, tVar = evaluation_des_performances( discretisation( Xtest,  K ), Ytest, modeles,  K )
            trainRes.append(aMean)
            testRes.append(tMean)
            trainVarRes.append(aVar)
            testVarRes.append(tVar)
        np.save("Res[N=%d,K=(5,200)]"%(N), [trainRes, testRes, trainVarRes, testVarRes])
            
if __name__ == "__main__":
    data, X, Y, nCl = load_data('TME6_lettres.pkl')
    main(data, X, Y, nCl)
