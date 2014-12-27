#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import methodes_discriminantes as md

def dessine_regression(vrais, nom_iter, epsilon):
    fig = plt.figure()
    x = range(nom_iter+1)
    y = vrais
    plt.plot( x, y, label="$\epsilon=%f$"%epsilon)
    plt.xlabel("Nombre d'Iteration")
    plt.ylabel("Log Vraisemblance")
    plt.legend()
    plt.savefig("regression(eps%f).png"%epsilon)
    plt.close(fig)

def evaluation_qualitative( X, Y, group, models):
    conf = np.zeros((10,10))
    for i, cls in enumerate(group):
        for echantillon in cls:
            conf[i, np.argmax([ f( X[echantillon], mdl[0], mdl[1] ) for mdl in models ]) ] += 1
    fig = plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")
    plt.close(fig)
    
def dessine(x,cname=None):
    # pour afficher le vecteur des moyennes des pixels pour le modèle 0:
    fig = plt.figure()
    plt.imshow(x.reshape(16,16), cmap = cm.Greys_r, interpolation='nearest')
    if cname == None:
        plt.show()
    else:
        plt.savefig(cname)
    plt.close(fig)
    
def dessine_modeles_discriminants(X,Y):
    nom_iter = 120
    epsilon  = .00005
    
    for chiffre in range(10):
        md.nombre_genere(X,Y,chiffre)

    models = md.apprendre_classifieurs( x, y, nom_iter, epsilon )
    for chiffre,ml in enumerate(models):
        dessine(ml[0], "modele_discriminant[%d]"%chiffre)

# dessine_modeles_discriminants(X,Y)
