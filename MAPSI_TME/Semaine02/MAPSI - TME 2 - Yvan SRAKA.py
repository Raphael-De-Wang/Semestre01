#! /usr/bin/env python


# --- Bibliothèques ---

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import pyAgrum as gum


# --- La planche de Galton ---

# -- 1. Loi de Bernoulli --

def bernoulli(p):
    x = np.random.rand()
    if x > p:
        return 0
    else:
        return 1


# -- 2. Loi binomiale --

def binomiale(n, p):
    x = 0
    for i in range(n):
        x = x + bernoulli(p)
    return x


# -- 3. Histogramme de la loi binomiale --

n = 15
tab = np.array([binomiale(n, 0.5) for i in range(1000)])

plt.hist(tab, 10)
plt.show()


# ---- Visualisation d'independances ---

# -- 1. Loi normale centree --

def normale(k, sigma):  # != numpy.random.normal
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    y = np.zeros(k)
    for i, x in enumerate(np.arange(-2 * sigma, 2 * sigma, 4 * sigma / k)):
        y[i] = 1 / (sigma * np.sqrt(2 * np.pi)) * \
            np.exp(-0.5 * ((x / sigma) ** 2))
    return y

plt.plot(normale(25, 5))
plt.show()


# -- 2. Distribution de probabilité affine --

def proba_affine(k, slope):
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    if abs(slope) > 2 / (k * k):
        raise ValueError('la pente est trop raide : pente max = ' +
                         str(2 / (k * k)))
    y = np.zeros(k)
    for i in np.arange(k):
        y[i] = 1 / k + (i - (k - 1) / 2) * slope
    return y

plt.plot(proba_affine(25, 0.002))
plt.show()


# -- 3. Distribution jointe --

PA = np.array([0.2, 0.7, 0.1])
PB = np.array([0.4, 0.4, 0.2])


def Pxy(PA, PB):
    return np.array([[i * j for i in PA] for j in PB])

# Pxy(PA, PB) = array([[0.08,  0.08,  0.04],
#                     [0.28,  0.28,  0.14],
#                     [0.04,  0.04,  0.02]])


# -- 4. Affichage de la distribution jointe --

def dessine(P_jointe):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-3, 3, P_jointe.shape[0])
    y = np.linspace(-3, 3, P_jointe.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show()

dessine(Pxy(normale(25, 5), proba_affine(25, 0.002)))


# --- Indépendances conditionnelles et consommation mémoire ---

# -- 1. Préparation de l'exercice : marginalisation --

def generateProbaIndep(coefs):
    n = 2 ** coefs.size
    x = np.ones(n)
    for i in np.arange(n):
        for j in np.arange(coefs.size):
            if i % (2 ** j):
                x[i] *= coefs[j]
            else:
                x[i] *= 1 - coefs[j]
    return x

P1 = np.array([0.05, 0.1, 0.15, 0.2, 0.02, 0.18, 0.13, 0.17])
P2 = generateProbaIndep(np.array([0.2, 0.3, 0.4]))


def project1Var(P, index):
    """
    supprime 1 variable d'une probabilité jointe

    Param P : une distribution de proba jointe sous forme d'un array à 1
       dimension ( toutes les variables aléatoires sont supposées binaires )
    Param index : représente la variable aléatoire à marginaliser
       (0 = 1ère variable, 1 = 2ème variable, etc).
    """
    length = 2 ** (index + 1)
    reste = 2 ** index
    vect = np.zeros(P.size / 2)
    for i in np.arange(P.size):
        j = np.floor(i / length) * length / 2 + (i % reste)
        vect[j] += P[i]
    return vect


def project(P, ind_to_remove):
    """
    Calcul de la projection d'une distribution de probas

    Param P une distribution de proba sous forme d'un array à 1 dimension
    Param ind_to_remove un array d'index representant les variables à
    supprimer. 0 = 1ère variable, 1 = 2ème var, etc.
    """
    v = P
    ind_to_remove.sort()
    for i in np.arange(ind_to_remove.size - 1, -1, -1):
        v = project1Var(v, ind_to_remove[i])
    return v


# -- 2. Préparation de l'exercice : restauration des indices --

def expanse1Var(P, index):
    """
    duplique une distribution de proba |X| fois, où X est une des variables
    aléatoires de la probabilité jointe P. Les variables étant supposées
    binaires, |X| = 2. La duplication se fait à l'index de la variable passé
    en argument.
    Par exemple, si P = [0,1,2,3] et index = 0, expanse1Var renverra
    [0,0,1,1,2,2,3,3]. Si index = 1, expanse1Var renverra [0,1,0,1,2,3,2,3].

    Param P : une distribution de proba sous forme d'un array à 1 dimension
    Param index : représente la variable à dupliquer (0 = 1ère variable,
       1 = 2ème variable, etc).
    """
    length = 2 ** (index + 1)
    reste = 2 ** index
    vect = np.zeros(P.size * 2)
    for i in np.arange(vect.size):
        j = np.floor(i / length) * length / 2 + (i % reste)
        vect[i] = P[j]
    return vect


def expanse(P, ind_to_add):
    """
    Expansion d'une probabilité projetée

    Param P une distribution de proba sous forme d'un array à 1 dimension
    Param ind_to_add un array d'index representant les variables permettant
    de dupliquer la proba P. 0 = 1ère variable, 1 = 2ème var, etc.
    """
    v = P
    ind_to_add.sort()
    for ind in np.arange(ind_to_add.size):
        v = expanse1Var(v, ind)
    return v


# -- 3. Probabilités conditionnelles --

def nb_vars(P):
    i = P.size
    nb = 0
    while i > 1:
        i /= 2
        nb += 1
    return nb


def proba_conditionnelle(P):
    n = nb_vars(P)
    x = np.zeros(n - 1)
    for i in np.arange(n - 1):
        x[i] = project(P, np.delete(np.arange(n - 1), i))[3]
    return x

#print(proba_conditionnelle(P1))
#print(proba_conditionnelle(P2))


# -- 4. Indépendances conditionnelles --

def is_indep(P, index, epsilon):
    n = nb_vars(P)
    X = np.zeros(n - 1)
    for i in np.arange(n - 1):
        if i != index:
            if proba_conditionnelle(P) - expanse(project(P, np.delete(np.arange(n - 1), np.array([i, index]))), np.delete(np.arange(n - 1), index))[3] > epsilon:
                return False
    return True

filename = "2014_tme2_asia.txt"
P = np.loadtxt(filename)

print(is_indep(P, 7, 1e-6))


# -- 5. Exploitation d'indépendances conditionnelles --

def find_indep(P, epsilon):
    n = nb_vars(P)
    ind_to_remove = np.array([index for index in np.arange(n) if is_indep(P, index, epsilon)])
    return np.array([n, project(P, ind_to_remove), np.delete(np.arange(n), ind_to_remove)])


# -- 6. Expression compacte d'une probabilité jointe --

def find_all_indep(P, epsilon):
    n = nb_vars(P)
    m = 0
    for i in np.arange(n):
        C = project(P, np.delete(np.arange(n), find_indep(P, epsilon)))
        m += C.size
        print("nombre de variables de la distribution P(X0,…,Xi) : %s" % n)
        print("nombre de variables de la probabilité conditionnelle compacte calculée : %s" % nb_vars(C))
    print("consommation mémoire totale de P(X0,…,Xn) : %s" % P.size)
    print("consommation mémoire totale des probabilités conditionnelles compactes : %s" % m)


# -- 7. Applications pratiques --

# chargement du fichier bif ou dsl
filename = "nom du fichier"
bn = gum.loadBN(filename)

# affichage de la taille des probabilités jointes compacte et non compacte
print(bn)
