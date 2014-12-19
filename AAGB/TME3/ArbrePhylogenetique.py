# -*- coding: utf-8 -*-
import numpy as np


#  ==== Variables globales : ====

# Les transcrits pour le gene MAPK8 :
Human       = ["abcd", "acde", "abcdeh","abcdefiIjnpq","abcdefiIjnpz", "abcdegiIjnpz","abcdegiIjnpq","abcdeIjnpz"]
Mouse       = ["qpnjIigedcba","qpnjIifedcba","zpnjIigedcba","zpnjIedcba","zpnjIifedcba"]
Xenopus     = ["rpljIigedcba","qpomjIigedcba"]
ZebrafishA  = ["qpnjIifedcba","qpnjIigedcba"]
ZebbrafishB = ["zpnjIigedcba", "qpnjIigedcba", "gedcba"]
FuguA       = ["abcdefijk","abcdeIjnpz","abcdegiIjnpq","abcdefiIjnpq"]
FuguB       = ["abcdeIjnpz","abcdegiIjnpq","abcdefiIjnpq"]
Drosophila  = ["pnjiIgedcba"]
Celegans    = ["stuabcdegiIjn", "uabcdegiIjn"]

# C est une matrice 3x3 qui permet de fixer un cout de changement d’état des exons.
C = 1 - np.eye(3,3)

# E est l’ensemble des exons observé pour MAPK8 :
E = ["a","b","c","d","e","f","g","h","i","I","j","k","l","m","n","o","p","q","z","r","s","t","u"]

# 1. Ecrire la fonction exonState qui prend une liste de transcrits et renvoie exS l’état de chaque exon de E dans cette liste.


# 2. Lors la remontée de l’algorithme de Sankoff.




