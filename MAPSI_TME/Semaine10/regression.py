# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---- Données jouets ----
# Dans un premier temps, générons des données jouets paramétriques:

# X tirage avec un simple rand()
def tirage_X(N):
    return np.random.rand(N)    

# y= ax + b + ϵ, ϵ~N(0,σ)
def _f(a,b,X):
    return a * X + b
    
def f(a, b, X, sig, N ):
    # normal(loc=0.0, scale=1.0, size=None)
    return _f(a,b,X) + np.random.normal(0, sig, N)

# Pour faire cet énoncé, je suis parti des valeurs suivantes:
a = 6.
b = -1.
c = 1
# N nombre de points à générer 
N = 100
sig = .4
v = [-0.2,1.2,-2,6]

# Vous utiliserez la fonction np.random.randn pour le tirage de données selon une loi normale
X = np.random.rand(N)

Y = f(a,b,X,sig,N)

def dessine_donnees(X,Y,v,show= True, save=True):
    fig = plt.figure()
    plt.axis(v)
    plt.plot( X, Y, 'ro')
    if show:
        plt.show()
    if save:
        plt.savefig("regression_donnee.png")
        
    return fig

# dessine_donnees(X,Y, v,show=False)

# ---- Validation des formules analytiques ----

# Nous avons vu deux types de résolutions analytique: 

# 1. Estimation de paramètres probabiliste
# Estimer les paramètres, faire passer les x dans le modèle et tracer la courbe associée.
def E(M):
    return np.mean(M)

def a_estime(X,Y,sig):
    [[CovX,CovXY],[CovXY,CovY]] = np.cov(X,Y)
    return CovXY/CovX

def b_estime(X,Y,sig):
    return E(Y) - a_estime(X,Y,sig)*E(X)

def tracer_courbe_esti(a, b, X,Y,v):
    ae = a_estime(X,Y,sig)
    be = b_estime(X,Y,sig)
    x = np.array(np.linspace(0,1,10))
    y = a * x + b
    ye= _f(ae,be,x)
    
    fig = plt.figure()
    plt.axis(v)
    plt.plot( X, Y, 'ro')
    plt.plot( x, y)
    plt.plot( x, ye, 'g')
    plt.show()
    plt.savefig("regression_tracer_courbe_esti.png")

# tracer_courbe_esti(a,b,X,Y,v)

# 2. Formulation au sens des moindres carrés
# Nous partons directement sur une écriture matricielle. Du coup, il est nécessaire de construire la matrice X

# usage de hstack pour mettre cote à cote deux matrices: ATTENTION, la fonction prend en argument un tuple
# obligation de faire un reshape sur x (il ne sait pas comment agréger une matrice et un vecteur)
def matriceX(x,N):
    return np.hstack((x.reshape(N,1),np.ones((N,1))))

# print matriceX(X,N)

# Il faut ensuite poser et résoudre un système d'équations linéaires de la forme Aw=B
def produit_matriciel(X, M):
    return X.T.dot(M)

def resoudre_equations_lineaires(X,Y,N,CALLBACK=matriceX):
    X_ = CALLBACK(X,N)
    A = produit_matriciel(X_, X_)
    B = produit_matriciel(X_, Y)
    return np.linalg.solve(A,B)

def tracer_courbe_formu(a,b,X,Y,v):
    [af,bf] = resoudre_equations_lineaires(X,Y,N)
    x = np.array(np.linspace(0,1,10))
    y = a * x + b
    yf= _f(af,bf,x)
    
    fig = plt.figure()
    plt.axis(v)
    plt.plot( X, Y, 'ro')
    plt.plot( x, y)
    plt.plot( x, yf, 'g')
    plt.show()
    plt.savefig("regression_tracer_courbe_formu.png")
    

# tracer_courbe_formu(a,b,X,Y,v)

# ---- Optimisation par descente de gradient ----
# Soit un problème avec des données (xi,yi)i=1,…,N, une fonction de décision/prédiction paramétrée par un vecteur w et une fonction de cout à optimiser C(w). Notre but est de trouver les paramètres w* minimisant la fonction de coût:
# pour se rappeler du w optimal
## wstar = np.linalg.solve(X.T.dot(X), X.T.dot(y))
wstar = resoudre_equations_lineaires(X,Y,N)

# l'algorithme de la descente de gradient est le suivant
def gradient_cout(X,y,w):
    return 2 * X.T.dot((X.dot(w) - y))

def descente_gradient(X,y,eps=5e-3,nIterations=30):
    
    # 1. w0←init par exemple : 0
    w = np.zeros(X.shape[1]) # init à 0
    allw = [w]
    # 2. boucle 
    for t in xrange(nIterations):
        # wt+1←wt−ϵ∇wC(w)
        w = w - eps*gradient_cout(X,y,w)
        allw.append(w)
    
    return np.array(allw)

# print descente_gradient(matriceX(X,N),Y)

# Tester différentes valeurs d'epsilon
print "TODO: Tester différentes valeurs d'epsilon"

# Tester différentes initialisations
print "TODO: Tester différentes initialisations"

# comparer les résultats théoriques (solution analytique) et par descente de gradient
print "TODO: comparer les résultats théoriques (solution analytique) et par descente de gradient"

# On s'intéresse ensuite à comprendre la descente de gradient dans l'espace des paramètres. Le code ci-dessous permet de:
# Tracer le cout pour un ensemble de paramètres
def tracer_gradient(X,y):
    # tracer de l'espace des couts
    ngrid = 20
    w1range = np.linspace(-0.5, 8, ngrid)
    w2range = np.linspace(-1.5, 1.5, ngrid)
    w1,w2 = np.meshgrid(w1range,w2range)
    
    cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-y)**2).sum()) for w1i in w1range] for w2j in w2range])
    allw = descente_gradient(X,Y)
    
    plt.figure()
    plt.contour(w1, w2, cost)
    plt.scatter(wstar[0], wstar[1],c='r')
    plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )
    plt.show()


# tracer_gradient(matriceX(X,N),Y)


def tracer_gradient_3D(X,y):
    ngrid = 20
    w1range = np.linspace(-0.5, 8, ngrid)
    w2range = np.linspace(-1.5, 1.5, ngrid)
    w1,w2 = np.meshgrid(w1range,w2range)
    
    cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-y)**2).sum()) for w1i in w1range] for w2j in w2range])
    allw = descente_gradient(X,Y)

    costPath = np.array([np.log(((X.dot(wtmp)-y)**2).sum()) for wtmp in allw])
    costOpt  = np.log(((X.dot(wstar)-y)**2).sum())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(w1, w2, cost, rstride = 1, cstride=1 )
    ax.scatter(wstar[0], wstar[1],costOpt, c='r')
    ax.plot(allw[:,0],allw[:,1],costPath, 'b+-' ,lw=2 )
    plt.show()

# tracer_gradient_3D(matriceX(X,N),Y)

# ---- Extension non-linéaire (solution analytique) ----
# Générer de nouvelles données non-linéaires selon la formule: yquad=ax2+bx+c+ϵ, ϵ∼(0,σ) avec les mêmes valeurs de paramètres que précédemment.
def _yquad(a, b, c, X):
    return a * (X ** 2) + b * X + c
    
def yquad(a, b, c, X, sig, N):
    return _yquad(a, b, c, X) + np.random.normal(0,sig,N)

Y = yquad(a,b,c,X,sig,N)
v = [-0.2,1.2,-2,7]
# dessine_donnees(X,Y,v)

# Estimation de a,b,c au sens de l'erreur des moindres carrés:
# 1. construire un nouveau Xe=[x2, x,1] en utilisant la méthode hstack vue dans la question précédente:
def matriceXe(x,N):
    return np.hstack(((x**2).reshape(N,1),x.reshape(N,1),np.ones((N,1))))

# 2. estimer un vecteur w sur ces données en utilisant la solution analytique (les données sont petites, pas la peine de s'embéter)
[a,b,c] = resoudre_equations_lineaires(X,Y,N,matriceXe)

# 3. reconstruire yquad
y = yquad(a, b, c, X, sig, N)

# 4. calculer l'erreur moyenne de reconstruction
print "l'erreur moyenne de reconstruction: ", np.mean(Y-y)

def dessine_non_lineaire(X,Y):
    [a,b,c] = resoudre_equations_lineaires(X,Y,N,matriceXe)
    x = np.linspace(0,1,100)
    y = _yquad(a,b,c,x)
    
    dessine_donnees(X,Y,v,False,False)
    plt.plot( x, y)
    plt.show()
    plt.savefig("regression_non_lineaire.png")

# dessine_non_lineaire(X,Y)
    
# ---- Données réelles ----
# Source originale: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

# ---- Chargement des données ----

# Le jeu de données sera séparé en:
# 1. un jeu d'apprentissage (pour l'estimation des paramètres)
# 2. un jeu de test (pour l'estimation non biaisée de la performance)

data = np.loadtxt("winequality-red.csv", delimiter=";", skiprows=1)
N,d = data.shape # extraction des dimensions
pcTrain  = 0.7 # 70% des données en apprentissage
allindex = np.random.permutation(N)
indTrain = allindex[:int(pcTrain*N)]
indTest = allindex[int(pcTrain*N):]
X = data[indTrain,:-1] # pas la dernière colonne (= note à prédire)
Y = data[indTrain,-1]  # dernière colonne (= note à prédire)
# Echantillon de test (pour la validation des résultats)
XT = data[indTest,:-1] # pas la dernière colonne (= note à prédire)
YT = data[indTest,-1]  # dernière colonne (= note à prédire)

'''
Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)
'''

# Tester différents modèles de régression sur ce jeu de données.
def modele(W,X,puis=None):
    if puis != None:
        return W.dot(X**puis) # Non-lineaire
    
    return W.dot(X) # lineaire

def matriceXp(X,puis=None):
    if puis != None:
        return np.hstack(((X**puis).T,np.ones((len(X),1))))
    
    return np.hstack((X,np.ones((len(X),1))))

puis = np.ones(len(X))
X_ = matriceXp(X,puis)
print X_
wstar = resoudre_equations_lineaires(X,Y,puis,matriceXp)
print wstar
print modele(wstar,X_)

## -- undone -- 
