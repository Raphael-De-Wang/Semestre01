#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
from scipy import stats

from scipy import optimize
from math import pow, log, exp
from random import random

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )  
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( map ( lambda x: float(x), champs ) )    
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output
    
def display_image ( X ):
    """
    Etant donné un tableau de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()
    
X=read_file ('2014_tme3_usps_train.txt')

def learnML_class_parameters ( classe ):
    a=X[classe]    
    vect = np.zeros ( len(X))
    for i in range (len(a)):
        mean=np.append(vect, numpy.mean(a[i],axis=0 ))
        std=np.append(vect, numpy.std(a[i], axis=0))
        variance=std*std
        ML=np.atleast_2d(mean, variance)
        return ML
        
# print learnML_class_parameters (9)

def learnML_all_parameters ( train_data):
    tab=[]    
    for i in range (len(train_data)):
            (mean, variance) = learnML_class_parameters ( i )
            tab += [(mean, variance)]
    return tab
    
parameters = learnML_all_parameters ( X)
# print parameters

Y=read_file ('2014_tme3_usps_test.txt')

def list_mu(train_data):
    tab1=[]
    for i in range (len(train_data)):
            (mean, variance) = learnML_class_parameters ( i )
            for j in range(len(mean)):
                tab1= np.append(tab1,mean[j])   
    return tab1

    
# print list_mu(X).size 

def list_sigma(train_data):
    tab2=[]
    for i in range (len(train_data)):
        (mean, variance) = learnML_class_parameters ( i )
        print 'mean.shape, variance.shape', mean.shape, variance.shape
        for j in range(len(variance)):
            tab2 = np.append(tab2,variance[j])

    return tab2

sigma= list_sigma(X) 
print sigma.size
display_image ( sigma[1] )
