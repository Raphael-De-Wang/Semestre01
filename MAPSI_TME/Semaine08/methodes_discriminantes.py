#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import pickle as pkl

# fonction de suppression des 0 (certaines variances sont nulles car les pixels valent tous la même chose)
def woZeros(x): 
    y = np.where(x==0., 1., x)
    return y

# Apprentissage d'un modèle naïf où chaque pixel est modélisé par une gaussienne (+hyp. d'indépendance des pixels)
# cette fonction donne 10 modèles (correspondant aux 10 classes de chiffres)
# USAGE: theta = learnGauss ( X,Y )
# theta[0] : modèle du premier chiffre,  theta[0][0] : vecteur des moyennes des pixels, theta[0][1] : vecteur des variances des pixels
def learnGauss (X,Y): 
    theta = [(X[Y==y].mean(0),woZeros(X[Y==y].var(0))) for y in np.unique(Y)]
    return (np.array(theta)) 

# Application de TOUS les modèles sur TOUTES les images: résultat = matrice (nbClasses x nbImages)
def logpobs(X, theta):
    logp = [[-0.5*np.log(mod[1,:] * (2 * np.pi )).sum() + -0.5 * ( ( x - mod[0,:] )**2 / mod[1,:] ).sum () for x in X] for mod in theta ]
    return np.array(logp)

######################################################################
#########################     script      ############################


# Données au format pickle: le fichier contient X, XT, Y et YT
# X et Y sont les données d'apprentissage; les matrices en T sont les données de test
data = pkl.load(file('usps_small.pkl','rb'))

X = data['X']
Y = data['Y']
XT = data['XT']
YT = data['YT']

theta = learnGauss ( X,Y ) # apprentissage

logp  = logpobs(X, theta)  # application des modèles sur les données d'apprentissage
logpT = logpobs(XT, theta) # application des modèles sur les données de test

ypred  = logp.argmax(0)    # indice de la plus grande proba (colonne par colonne) = prédiction
ypredT = logpT.argmax(0)

# print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
# print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()

(I,J) = np.shape(X)
NOMBRE = 0
y = np.where(Y==NOMBRE,1.,0.)

def init_w0_b0():
    return (np.random.rand(256), np.random.rand())

w,b = init_w0_b0()
    
def f(x, w, b) :
    return 1/(1+ np.exp(-x.dot(w) -b ))

print f(x, 0, 0)
    
def log_vraisemblance( x, y, w, b):
    lv = y * np.log(f(x,w,b)) + (1-y) * np.log(1-f(x,w,b))
    return np.minimum(lv[ lv != np.NINF ],1-1e-8)

# print log_vraisemblance( X, y, w, b)

def derive_w(x, j, y, w, b) :
    return ( x[:, j]* (y - f(x,w,b))).sum()
    
def derive_b(x, y, w, b) :
    return (y - f(x,w,b)).sum()

def regression_logistique( X, Y, w, b, epsilon, N):
    lv = [ log_vraisemblance( X, y, w, b) ]
    for i in range(N):
        w = [ w[j] + epsilon * derive_w(X, j, Y, w, b) for j in range(256) ]
        b = b + epsilon * derive_b(X, y, w, b)
        lv.append(log_vraisemblance( X, y, w, b))
    return lv#x, w, b
    
# print regression_logistique( X, y, w, b, .001, 3)


