#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import heapq
import numpy as np
import pickle as pkl
import os.path as op
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
def extraire_data():
    data = pkl.load(file('usps_small.pkl','rb'))
    
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    XT = np.array(data['XT'])
    YT = np.array(data['YT'])

    theta = learnGauss ( X,Y ) # apprentissage
    return ( X,Y,XT,YT,theta )

def taux_bonne_classification(X,Y,XT,YT,theta):
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
        
    for i in xrange(N):
        w = [ w[j] + epsilon * derive_w(X, j, y, w, b) for j in range(256) ]
        b = b + epsilon * derive_b(X, y, w, b)
        if lv != None:
            lv.append(log_vraisemblance ( X, y, w, b) )
        
    return [ w, b ]

def modele_binaire_apprentissage(X,Y):
    vrais   = []
    NOMBRE  = 0
    nom_iter= 3
    epsilon = .00005
    (I,J)   = np.shape(X)
    w,b     = init_w0_b0(J)
    y       = class_vecteur(Y,NOMBRE)
    
    regression_logistique( X, y, w, b, epsilon, nom_iter, vrais)
    return (vrais, nom_iter, epsilon)
    # dessine_regression(vrais, nom_iter, epsilon)

# modele_binaire_apprentissage(X,Y)

#### Paradigme un-contre-tous pour le passage au multi-classe
def apprendre_classifieurs( X, Y, nom_iter=120, epsilon=.00005):
    (I,J)   = np.shape(X)
    thetaRL = []
    w,b     = init_w0_b0(J)
    for NOMBRE in range(len(np.unique(Y))):
        # Pour transformer le vecteur Y afin de traiter la classe NOMBRE:
        print "NOMBRE: ", NOMBRE
        print "w[0], b", w[0], b
        Yc      = class_vecteur(Y,NOMBRE)
        thetaRL.append(regression_logistique( X, Yc, w, b, epsilon, nom_iter))

    return thetaRL

def groupByLabel( y ) :
    index = []
    for i in np.unique( y ): # pour toutes les classes
        ind, = np.where( y == i )
        index.append( ind )
    return index

def evaluation_qualitative( X, Y, group, models,label=""):
    conf = np.zeros((10,10))
    for i, cls in enumerate(group):
        for echantillon in cls:
            conf[i, np.argmax([ f( X[echantillon], mdl[0], mdl[1] ) for mdl in models ]) ] += 1
    fig = plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(10),np.unique(Y))
    plt.yticks(np.arange(10),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("evaluation_qualitative[%s].png"%label)
    plt.close(fig)
    
def un_contre_tous(X,Y,XT,YT, thetaRL,label=''):
    # nom_iter= 3
    # epsilon = .00005
    # thetaRL = apprendre_classifieurs(X,Y,nom_iter,epsilon)
    
    pRL  = np.array([[ f(x,mod[0],mod[1]) for x in X ] for mod in thetaRL ])
    pRLT = np.array([[ f(x,mod[0],mod[1]) for x in XT] for mod in thetaRL ])
    
    ypred  = pRL.argmax(0)
    ypredT = pRLT.argmax(0)
    
    print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
    print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()
    
    # taux_bonne_classification(X,Y,XT,YT,theta)
    
    # group = groupByLabel(YT)
    # evaluation_qualitative( XT, YT, group, thetaRL, label)
    
    return np.where(ypred != Y, 0.,1.).mean(),np.where(ypredT != YT, 0.,1.).mean()
    
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

def nombre_genere(X,Y,nombre):
    mu    = learn_mu(X,Y,nombre)
    var   = learn_std(X,Y,nombre)
    x     = tirer_un_nombre(var, mu)
    return (x,nombre)
    
def modele_discriminant(X, Y, nombre, nom_iter=120, epsilon=.00005):
    y = Y[Y==nombre]
    x = X[Y==nombre]
    thetaRL = apprendre_classifieurs( x, y, nom_iter, epsilon)
    # dessine(np.array(thetaRL[0][0]),"%s_discriminant"%nombre)
    return thetaRL[0]

# modele_discriminant(X, Y, 2)

#### Optimisation du modèle discriminant
def regression_logistique_optimal( X, y, w, b, lv, epsilon, gd, iter_borne = 200):
    # Mettre en place un critère d'arrêt basé sur l'évolution de la vraisemblance dans la procédure évolutive. L'idée est la même que dans les séances passée
    i = 0
    lv.append(log_vraisemblance ( X, y, w, b) )
    while True:
        w = [ w[j] + epsilon * derive_w(X, j, y, w, b) for j in range(256) ]
        b = b + epsilon * derive_b(X, y, w, b)
        lv.append(log_vraisemblance ( X, y, w, b))
        i += 1
        print "lv[i] - lv[i-1] = ", lv[i] - lv[i-1] 
        if ( lv[i] - lv[i-1] < gd ) or ( i > iter_borne ) :
            break

    return ( w, b )

def modele_discriminant_optimisation(X,Y,lv,nombre,epsilon=.00005,gd=0.01):
    (I,J)   = np.shape(X)
    w,b     = init_w0_b0(J)
    Yc      = class_vecteur(Y,nombre)
    return regression_logistique_optimal( X, Yc, w, b, lv, epsilon, gd)

# Mettre en place une stratégie de rejet des échantillons ambigus: étudier l'amélioration des résultats en fonction du nombre d'échantillon rejetés. Etudier en particulier les cas suivants:
def passe_borne(X, Y, models, seuil):
    p = np.array([ max([ f( X[i], mdl[0], mdl[1] ) for mdl in models ]) for i in range(len(X)) ])
    return X[p>seuil], Y[p>seuil]

def ambigus_proche(X, Y, models, seuil):
    p = []
    for i in range(len(X)):
        vrais = heapq.nlargest(2, [ f( X[i], mdl[0], mdl[1] ) for mdl in models ] )
        p.append(vrais[0] - vrais[1])
    p = np.array(p)
    return X[p>seuil], Y[p>seuil]

#### Rapport 02 ####
'''
models  = apprendre_classifieurs( X, Y )
np.save("models", models)
'''

def evaluer_passe_borne(X,Y,models):
    for seuil in np.linspace(0,1,11):
        print "---- passe_borne Seuil = %f ----"%seuil
        if op.exists("models[passe_borne,seuil=%f].npy"%seuil):
            continue
        (Xp,Yp) = passe_borne(X,Y,models,seuil)
        if len(Y) == len(Yp):
            continue
        if len(X) == len(Xp):
            continue
        if len(np.unique(Y)) != len(np.unique(Yp)):
            continue
        models2 = apprendre_classifieurs( Xp, Yp )
        np.save("models[passe_borne,seuil=%f]"%seuil,models2)

def evaluer_ambigus_proche(X,Y,models):
    for seuil in np.linspace(1,0,11):
        print "---- ambigus_proche Seuil = %f ----"%seuil
        if op.exists("models[ambigus_proche,seuil=%f].npy"%seuil):
            continue
        (Xa,Ya) = ambigus_proche(X,Y,models,seuil)
        if len(Y) == len(Ya):
            continue
        if len(X) == len(Xa):
            continue
        if len(np.unique(Y)) != len(np.unique(Ya)):
            continue
        models3 = apprendre_classifieurs( Xa, Ya )
        np.save("models[ambigus_proche,seuil=%f]"%seuil,models3)
    
def evaluer_passe_borne_modeles():
    aList = []
    tList = []
    sList = []
    for seuil in np.linspace(0,1,11):
        if op.exists("models[passe_borne,seuil=%f].npy"%seuil):
            print "---- passe_borne Seuil = %f ----"%seuil
            models_load = np.load("models[passe_borne,seuil=%f].npy"%seuil)
            aMean, tMean = un_contre_tous(X,Y,XT,YT, models_load)
            aList.append(aMean)
            tList.append(tMean)
            sList.append(seuil)            
            # group = groupByLabel(YT)
            # evaluation_qualitative( XT, YT, group, models_load, "passe_borne,seuil=%f"%seuil)
            
    return ( aList, tList, sList)
    
def evaluer_ambigus_proche_modeles():
    aList = []
    tList = []
    sList = []
    for seuil in np.linspace(0,1,11):
        if op.exists("models[ambigus_proche,seuil=%f].npy"%seuil):
            print "---- ambigus_proche Seuil = %f ----"%seuil
            models_load = np.load("models[ambigus_proche,seuil=%f].npy"%seuil)
            aMean, tMean = un_contre_tous(X,Y,XT,YT, models_load)
            aList.append(aMean)
            tList.append(tMean)
            sList.append(seuil)
            # group = groupByLabel(YT)
            # evaluation_qualitative( XT, YT, group, models_load, "ambigus_proche,seuil=%f"%seuil)
            
    return ( aList, tList, sList )    

def trace_comp(aStd, tStd, trainRes, testRes, xList, x_borne=None, y_borne=None):
    fig = plt.figure()
    if len(trainRes) != len(testRes):
        raise("InvalidDataSet")
    if x_borne != None:
        plt.xlim(x_borne[0],x_borne[1])
    if y_borne != None:
        plt.xlim(y_borne[0],y_borne[1])
    plt.plot(xList, [aStd for i in xrange(len(xList))], label='Train[Original]')
    plt.plot(xList, [tStd for i in xrange(len(xList))], label='Test[Original]')
    plt.plot(xList, trainRes, label='Train[passe_borne]')
    plt.plot(xList, testRes,  label='Test[passe_borne]')
    plt.legend()
    plt.savefig("comparation_performance_train_test.png")
    plt.close(fig)
    
def dessine_modeles_discriminants(X,Y, models):
    nom_iter = 120
    epsilon  = .00005
    
    for chiffre in range(10):
        x,nombre = nombre_genere(X,Y,chiffre)
        dessine(x,"%s_generatif"%nombre)
        
    for chiffre,ml in enumerate(models):
        dessine(np.array(ml[0]), "modele_discriminant[%d]"%chiffre)

def dessine(x,cname=None):
    # pour afficher le vecteur des moyennes des pixels pour le modèle 0:
    fig = plt.figure()
    plt.imshow(x.reshape(16,16), cmap = cm.Greys_r, interpolation='nearest')
    if cname == None:
        plt.show()
    else:
        plt.savefig(cname)
    plt.close(fig)
        
models = np.load("models.npy")
X,Y,XT,YT,theta = extraire_data()

# evaluer_passe_borne(X,Y,models)
# evaluer_ambigus_proche(X,Y,models)
group = groupByLabel(YT)
# evaluation_qualitative( XT, YT, group, models, "models")

aMean, tMean = un_contre_tous(X,Y,XT,YT, models)
aList, tList, sList =  evaluer_ambigus_proche_modeles()
# aList, tList, sList =  evaluer_passe_borne_modeles()
trace_comp(aMean, tMean, aList, tList, sList)
# dessine_modeles_discriminants(X,Y, models)

