#!/usr/bin/env python


# --- Bibliothèques ---

import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl


# --- Récupération des données ---

# Solution 1: Télécharger un fichier

fname = "dataVelib.pkl"
f = open(fname, 'rb')
data = pkl.load(f)
f.close()


# --- Mise en forme et élimination du bruit ---

m0 = np.array([[station['position']['lat'], station['position']['lng'], station['alt'], station['number'] // 1000, station['bike_stands'], station['available_bike_stands']] for station in data if 1 <= station['number'] // 1000 and station['number'] // 1000 <= 20])


# --- Distributions de probabilités ---

Ar = np.array(m0[:, 3], dtype=np.int16)  # Arrondissement
Al = np.array(m0[:, 2], dtype=np.float64)  # Altitude
# Station pleine [variable binaire: valeur 1 si la station est pleine]
Sp = np.where(m0[:, 5] <= 0, 1, 0)
# Au moins 2 vélos disponibles (variable binaire)
Vd = np.where(m0[:, 5] >= 2, 1, 0)

# P[Ar]
pAr = np.zeros((20), dtype=np.float64)
for ar in Ar:
    pAr[ar - 1] += 1
pAr /= Ar.size

# P[Al]
r = plt.hist(Al, 30)
pAl = r[0] / r[0].sum()
pAl /= r[1][1] - r[1][0]


# --- Tracer un histogramme ---

# Tracer l'histogramme correspondant à la distribution P[Al]

plt.bar((r[1][1:] + r[1][:-1]) / 2, pAl, r[1][1] - r[1][0])
plt.savefig("Figure 0.pdf")


# --- Calcul de probabilités conditionnelles ---

# [0,...,29] catégorie d'altitude
cAl = np.floor((m0[:, 3] - m0[:, 3].min()) / (r[1][1] - r[1][0]))

# P[Vd|Al]
pVdAl = np.zeros((2, np.unique(cAl).shape[0]))

for al in np.unique(cAl):
    n, = np.where(cAl == al)
    for i in n:
        if Vd[i] == 1:
            pVdAl[1][al] += 1
        else:
            pVdAl[0][al] += 1

# E[P[Vd|Al]]
EpVdAl = pVdAl[1].sum() / pVdAl.sum()

# P[Sp|Al]
pSpAl = np.zeros((2, np.unique(cAl).shape[0]))

for al in np.unique(cAl):
    n, = np.where(cAl == al)
    for i in n:
        if Sp[i] == 1:
            pSpAl[1][al] += 1
        else:
            pSpAl[0][al] += 1

# P[Vd|Ar]
pVdAl = np.zeros((2, np.arange(1, 21).shape[0]))

for ar in np.arange(1, 21):
    n, = np.where(Ar == ar)
    for i in n:
        if Vd[i] == 1:
            pVdAl[1][al] += 1
        else:
            pVdAl[0][al] += 1


# --- Tracer la population des stations ---

x1 = m0[:, 1]  # -> longitude
x2 = m0[:, 0]  # -> latitude

# -- Sanity check --

# Affichage des arrondissements pour vérifier que tout est OK

style = [(s, c) for s in "o^+*" for c in "byrmck"]

plt.figure()

for ar in np.arange(1, 21):
    i, = np.where(Ar == ar)
    plt.scatter(x1[i], x2[i], marker=style[ar - 1][0],
                c=style[ar - 1][1], linewidths=0)

plt.axis('equal')
plt.legend(np.arange(1, 21), fontsize=10)

plt.savefig("Figure 1.pdf")

# -- Disponibilité --

# Projeter les stations sur la carte en mettant en:
# - rouge les stations pleines,
# - jaune les vides,
# - verte les autres.

plt.figure()

yellow, = np.where((0 < m0[:, 5]) & (m0[:, 5] < m0[:, 4]))
plt.scatter(x1[yellow], x2[yellow], marker="^", c="y", linewidths=0)

red, = np.where(0 == m0[:, 5])
plt.scatter(x1[red], x2[red], marker="^", c="r", linewidths=0)

green, = np.where(m0[:, 5] == m0[:, 4])
plt.scatter(x1[green], x2[green], marker="^", c="g", linewidths=0)

plt.axis('equal')
plt.legend(["Pleines", "Vides", "Autres"], fontsize=10)

plt.savefig("Figure 2.pdf")

# -- Moyenne, Médiane ---

# Stations dont l'altitude est inférieure à la moyenne

plt.figure()

for ar in np.arange(1, 21):
    i, = np.where((Ar == ar) & (Al < np.mean(Al)))
    plt.scatter(x1[i], x2[i], marker=style[ar - 1][0],
                c=style[ar - 1][1], linewidths=0)

plt.axis('equal')
plt.legend(np.arange(1, 21), fontsize=10)

plt.savefig("Figure 3.pdf")

# Stations dont l'altitude est supérieure à la médiane

plt.figure()

for ar in np.arange(1, 21):
    i, = np.where((Ar == ar) & (Al > np.median(Al)))
    plt.scatter(x1[i], x2[i], marker=style[ar - 1][0],
                c=style[ar - 1][1], linewidths=0)

plt.axis('equal')
plt.legend(np.arange(1, 21), fontsize=10)

plt.savefig("Figure 4.pdf")

# -- Tests de corrélation --

# Calculer les corrélations entre les variables Altitude et Vélo disponible
x1 = np.correlate(Al, Vd)

# Calculer les corrélations entre les variables Altitude et Vélo disponible
x2 - np.correlate(Ar, Vd)

# Quel facteur est le plus lié au fait qu'(au moins) un vélo soit disponible dans une station ?
# Réponse : L'altitude !
