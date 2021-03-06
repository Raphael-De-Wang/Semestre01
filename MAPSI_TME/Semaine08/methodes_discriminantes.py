#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import heapq
import numpy as np
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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

print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()

#### Approche discriminante
def class_vecteur(Y,nombre):
    return np.where(Y==nombre,1.,0.)

def init_w0_b0(J):
    return (np.random.rand(J)/10, np.random.rand())

def f(x, w, b) :
    return  1/(1+ np.exp(-x.dot(w) -b ))

def log_vraisemblance( x, y, w, b):
    return (y * np.log(f(x,w,b)) + (1-y) * np.log(1-f(x,w,b))).sum()

def derive_w(x, j, y, w, b) :
    return (x[:, j]* (y - f(x,w,b))).sum()

def derive_b(x, y, w, b) :
    return (y - f(x,w,b)).sum()

#### Apprentissage d'un modèle binaire
def regression_logistique( X, y, w, b, epsilon, N, lv=None ):
    
    if lv != None:
        lv.append(log_vraisemblance ( X, y, w, b) )
        
    for i in range(N):
        w = [ w[j] + epsilon * derive_w(X, j, y, w, b) for j in range(256) ]
        b = b + epsilon * derive_b(X, y, w, b)
        if lv != None:
            lv.append(log_vraisemblance ( X, y, w, b) )
        
    return ( w, b )

def dessine_regression(vrais, nom_iter, epsilon):
    fig = plt.figure()
    x = range(nom_iter+1)
    y = vrais
    plt.plot( x, y, label="$\epsilon=%f$"%epsilon)
    plt.xlabel("Nombre d'Iteration")
    plt.ylabel("Log Vraisemblance")
    plt.legend()
    plt.savefig("regression(eps%f).png"%epsilon)

def modele_binaire_apprentissage(X,Y):
    vrais   = []
    NOMBRE  = 0
    nom_iter= 3
    epsilon = .00005
    (I,J)   = np.shape(X)
    w,b     = init_w0_b0(J)
    y       = class_vecteur(Y,NOMBRE)
    
    regression_logistique( X, y, w, b, epsilon, nom_iter, vrais)
    dessine_regression(vrais, nom_iter, epsilon)

# modele_binaire_apprentissage(X,Y)

#### Paradigme un-contre-tous pour le passage au multi-classe
def apprendre_classifieurs( X, Y, nom_iter, epsilon):
    (I,J)   = np.shape(X)
    thetaRL = []
    for NOMBRE in range(len(np.unique(Y))):
        # Pour transformer le vecteur Y afin de traiter la classe NOMBRE:
        Yc      = class_vecteur(Y,NOMBRE)
        w,b     = init_w0_b0(J)
        thetaRL.append(regression_logistique( X, Yc, w, b, epsilon, nom_iter))

    return thetaRL

def groupByLabel( y ) :
    index = []
    for i in np.unique( y ): # pour toutes les classes
        ind, = np.where( y == i )
        index.append( ind )
    return index

def evaluation_qualitative( X, Y, group, models):
    conf = np.zeros((10,10))
    for i, cls in enumerate(group):
        for echantillon in cls:
            conf[i, np.argmax([ f( X[echantillon], mdl[0], mdl[1] ) for mdl in models ]) ] += 1
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")
    
def un_contre_tous(X,Y,XT,YT):
    nom_iter= 3
    epsilon = .00005
    thetaRL = apprendre_classifieurs(X,Y,nom_iter,epsilon)
    
    pRL  = np.array([[ f(x,mod[0],mod[1]) for x in X ] for mod in thetaRL ])
    pRLT = np.array([[ f(x,mod[0],mod[1]) for x in XT] for mod in thetaRL ])
    
    ypred  = pRL.argmax(0)
    ypredT = pRLT.argmax(0)
    
    print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
    print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()
    
    group = groupByLabel(Y)
    evaluation_qualitative( X, Y, group, thetaRL)
    
# un_contre_tous(X,Y,XT,YT)

#### Analyse qualitative comparée des modèles (génératifs/discriminants)

def learn_mu(X,Y,nombre):
    return np.mean(X[Y==nombre],0)

def learn_std(X,Y,nombre):
    var = np.std(X[Y==nombre],0)
    # Note 2: dans un premier temps, nous avions détecté les variances nulles pour les mettre à 1, Mais cette aménagement doit être supprimé lors de la génération de chiffre.
    var[var==0] = 1
    return var

def tirer_un_nombre(sigma, mu):
    x = np.random.randn(1) * sigma + mu
    # Note 3: les pixels générés peuvent être positifs... Ou négatifs. Alors que dans la base de données, tous les pixels sont positifs... Du coup, pour obtenir un résultat plus vraisemblable, il est intéressant de ne garder que les valeurs positives (et de mettre à 0 les valeurs négatives).
    x[ x < 0 ] = 0
    return x

def dessine(x):
    # pour afficher le vecteur des moyennes des pixels pour le modèle 0:
    plt.figure()
    plt.imshow(x.reshape(16,16), cmap = cm.Greys_r, interpolation='nearest')
    plt.show()

def nombre_genere(X,Y):
    nombre= 2
    mu    = learn_mu(X,Y,nombre)
    var   = learn_std(X,Y,nombre)
    x     = tirer_un_nombre(var, mu)
    dessine(x)
    
def modele_discriminant(X, Y, nombre):
    nom_iter = 120
    epsilon  = .00005
    y = Y[Y==nombre]
    x = X[Y==nombre]
    thetaRL = apprendre_classifieurs( x, y, nom_iter, epsilon)
    dessine(thetaRL[0][0])

# modele_discriminant(X, Y, 2)

#### Optimisation du modèle discriminant
def regression_logistique_optimal( X, y, w, b, lv, epsilon, gd, iter_borne = 200):
    # Mettre en place un critère d'arrêt basé sur l'évolution de la vraisemblance dans la procédure évolutive. L'idée est la même que dans les séances passée
    i = 0
    lv.append(log_vraisemblance ( X, y, w, b) )
    while True:
        w = [ w[j] + epsilon * derive_w(X, j, y, w, b) for j in range(256) ]
        b = b + epsilon * derive_b(X, y, w, b)
        lv.append(log_vraisemblance ( X, y, w, b) )
        i += 1
        if ( lv[i] - lv[i-1] < gd ) or ( i > iter_borne ) :
            break

    return ( w, b )

def modele_discriminant_optimisation(X,Y,lv,nombre):
    gd      = 0.01
    epsilon = .00005
    (I,J)   = np.shape(X)
    w,b     = init_w0_b0(J)
    Yc      = class_vecteur(Y,nombre)
    return regression_logistique_optimal( X, Yc, w, b, lv, epsilon, gd)

# Mettre en place une stratégie de rejet des échantillons ambigus: étudier l'amélioration des résultats en fonction du nombre d'échantillon rejetés. Etudier en particulier les cas suivants:
def rejet_echantillons_ambigus(X,Y,models,seuil,CALLBACK):
    return X[CALLBACK(X, models, seuil)], Y[CALLBACK(X, models, seuil)]

def passe_borne(X, models, seuil):
    return [ max([ f( X[i], mdl[0], mdl[1] ) for mdl in models ]) > seuil for i in range(len(X)) ]

def ambigus_proche(X, models, seuil):
    p = []
    for i in range(len(X)):
        vrais = heapq.nlargest(1, [ f( X[i], mdl[0], mdl[1] ) for mdl in models ] )
        p.append(vrais[0] - vrais[1] < seuil)
    return p


