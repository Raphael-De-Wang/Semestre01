# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
def load_data(fname):
    data = pkl.load(file( fname, "rb"))
    X = np.array(data.get('letters'))
    Y = np.array(data.get('labels'))
    nCl = 26
    return data, X, Y, nCl
    
data, X, Y, nCl = load_data('TME6_lettres.pkl')


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
def learnHMM(allx, allq, N, K, initTo0=False):
    
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
            alpha.append((Pi,A,B))
            # 2. Estimation des états cachés par Viterbi
            for [i] in np.argwhere(Y==lettre):
                ( p_est, s_est ) = viterbi(Xd[i],Pi,A,B)
                q[i] = s_est
                lv  += p_est
                
        lv_lst.append(lv)
                
    return alpha

def tracer_evolution_vraisemblance(lv_list):
    fig = plt.figure()
    x = range(len(lv_list))
    y = lv_list
    plt.plot( x, y )
    plt.xlabel("Nombre d'Iteration")
    plt.ylabel("Log Vraisemblance")
    plt.savefig("vraisemblance_regression.png")
    
'''
lv_lst = []
baum_welch_simplifie( lv_lst, X, Y,)
tracer_evolution_vraisemblance(lv_lst)
'''

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
def evaluation_qualitative( X, Y, group, models, N = 5, K = 10):
    conf = np.zeros((26,26))
    Xd   = discretisation(X,K)
    for i, cls in enumerate(group):
        for echantillon in cls:
            conf[i, np.argmax([ viterbi( Xd[echantillon], mdl[0], mdl[1], mdl[2] )[0] for mdl in models ]) ] += 1
                
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("evaluation.png")
    
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

# affichage d'une lettre (= vérification bon chargement)
def tracerLettre(let):
    a = -let*np.pi/180;
    coord = np.array([[0, 0]]);
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.plot(coord[:,0],coord[:,1])

# Faire tourner le code suivant qui réalise la génération de n échantillon pour nClred classes de lettres
def test( X, Y, models, N = 5, K = 10):
    #Trois lettres générées pour 5 classes (A -> E)
    n      = 3          # nb d'échantillon par classe
    nClred = 5   # nb de classes à considérer
    d      = K
    Xd     = discretisation(X,K)
    itrain = groupByLabel(Y)
    fig    = plt.figure()
    
    for cl in xrange(nClred):
        Pic = models[cl][0].cumsum() # calcul des sommes cumulées pour gagner du temps
        Ac  = models[cl][1].cumsum(1)
        Bc  = models[cl][2].cumsum(1)
        longeur = np.floor(np.array([len(x) for x in Xd[itrain[cl]]]).mean()) # longueur de seq. à générer = moyenne des observations
        for im in range(n):
            s,x = generateHMM(Pic, Ac, Bc, int(longeur))
            intervalle = 360./d  # pour passer des états => angles
            newa_continu = np.array([i*intervalle for i in x]) # conv int => double
            sfig = plt.subplot(nClred,n,im+n*cl+1)
            sfig.axes.get_xaxis().set_visible(False)
            sfig.axes.get_yaxis().set_visible(False)
            tracerLettre(newa_continu)
            plt.savefig("lettres_hmm.png")

lv_lst = []
models = baum_welch_simplifie( lv_lst, X, Y,)
# evaluation_qualitative( X, Y, groupByLabel(Y), models)
test(X, Y, models)
