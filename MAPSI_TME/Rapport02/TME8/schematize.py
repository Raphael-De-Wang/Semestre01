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

def dessine_modeles_discriminants(X,Y):
    nom_iter = 120
    epsilon  = .00005
    
    for chiffre in range(10):
        md.nombre_genere(X,Y,chiffre)

    models = md.apprendre_classifieurs( x, y, nom_iter, epsilon )
    for chiffre,ml in enumerate(models):
        dessine(ml[0], "modele_discriminant[%d]"%chiffre)

# dessine_modeles_discriminants(X,Y)
models = np.load("models.npy")
evaluation_qualitative( X, Y, group, models)
