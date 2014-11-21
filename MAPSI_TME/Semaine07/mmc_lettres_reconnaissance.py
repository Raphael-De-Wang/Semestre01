# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')
    
def load_data(fname):
    data = pkl.load(file( fname, "rb"))
    X = np.array(data.get('letters'))
    Y = np.array(data.get('labels'))
    nCl = 26
    return data, X, Y, nCl
    
data, X, Y, nCl = load_data('TME6_lettres.pkl')

def discretisation( X, n_etats = 10 ) :
    intervalle = 360. / n_etats
    return np.array([ np.floor( x / intervalle ) for x in X ])

def seq_Xd(Xd):
    return np.floor(np.linspace(0,N-.00000001,len(Xd)))
    
def initGD(X,N):
    return np.array([ seq_Xd(x) for x in X ])

N = 5
K = 10
Xd = discretisation(X,K)
q  = initGD(X,N)

def learnHMM(allx, allq, N, K, initTo0=False):
    
    if np.shape(allx) <> np.shape(allq):
        raise Exception("Invalid Data")
        
    if initTo0:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps

    for i in range(len(allx)):
        lettre   = allx[i]
        etat_seq = allq[i]
        Pi[etat_seq[0]] += 1
        for j in range( len(lettre) - 1 ):
            begin = etat_seq[j]
            to    = etat_seq[j+1]
            A[begin,to] += 1
            n = etat_seq[j]
            k = lettre[j]
            B[n,k] += 1
            
    A /= np.maximum(A.sum(1).reshape(N,1),1)
    B /= np.maximum(B.sum(1).reshape(N,1),1)
    Pi/= Pi.sum()
    return Pi, A, B

Pi, A, B = learnHMM( Xd[Y=='a'], q[Y=='a'], N, K)

def delta(x,Pi,A,B):
    if len(x) == 1:
        return np.log(Pi) + np.log(B[:,x[0]]), [[ -1 for i in range(len(Pi)) ]]
        
    dt_1, phi_table = delta(x[:-1],Pi,A,B)
    i = np.argmax(dt_1)
    
    return np.max(dt_1[i] + np.log(A[i])) + np.log(B[:,x[-1]]) , [] # phi_table.append(phi(dt_1,A,i))

def phi(delt_1,A,i):
    return np.argmax(delt_1 + np.log(A[i]))

def chemin(delt_list, S):
    return 0
    
def viterbi(x,Pi,A,B):
    # 2. Récursion:
    delta_t, phi_table = delta(x,Pi,A,B)
    
    # 3. Terminasion:
    p_est = max(delta_t)

    # 4. Chemin
    #    print phi_list
    # s_est = chemin(delt_list, S)
    return p_est

print viterbi(Xd[0],Pi,A,B)
    
'''
def calc_log_pobs_v2(x,Pi, A, B):
    
def baum_welch_simplifie():

def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

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
    return

#Trois lettres générées pour 5 classes (A -> E)
n = 3          # nb d'échantillon par classe
nClred = 5   # nb de classes à considérer
fig = plt.figure()
for cl in xrange(nClred):
    Pic = models[cl][0].cumsum() # calcul des sommes cumulées pour gagner du temps
    Ac = models[cl][1].cumsum(1)
    Bc = models[cl][2].cumsum(1)
    long = np.floor(np.array([len(x) for x in Xd[itrain[cl]]]).mean()) # longueur de seq. à générer = moyenne des observations
    for im in range(n):
        s,x = generateHMM(Pic, Ac, Bc, int(long))
        intervalle = 360./d  # pour passer des états => angles
        newa_continu = np.array([i*intervalle for i in x]) # conv int => double
        sfig = plt.subplot(nClred,n,im+n*cl+1)
        sfig.axes.get_xaxis().set_visible(False)
        sfig.axes.get_yaxis().set_visible(False)
        tracerLettre(newa_continu)
plt.savefig("lettres_hmm.png")

'''
