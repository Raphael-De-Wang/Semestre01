# -*- coding: utf-8 -*-

import math
import pydot        
import numpy as np
import pyAgrum as gum
import scipy.stats as stats
import gumLib.notebook as gnb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from gumLib.pretty_print import pretty_cpt

style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }

# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data ( data ) :
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1 # - nom variable
    res_data = np.zeros ( (nb_variables, nb_observations ), int )
    res_dico = np.empty ( nb_variables, dtype=object )

    # pour chaque variable, faire la traduction
    for i in range ( nb_variables ):
        res_dico[i] = {}
        index = 0
        for j in range ( 1, nb_observations + 1 ):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i,j] not in res_dico[i]:
                res_dico[i].update ( { data[i,j] : index } )
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i,j-1] = res_dico[i][data[i,j]]
    return ( res_data, res_dico )

# fonction pour lire les données de la base d'apprentissage
def read_csv ( filename ) :
    data = np.loadtxt ( filename, delimiter=',', dtype='string' ).T
    names = data[:,0].copy ()
    data, dico = translate_data ( data )
    return names, data, dico

# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table ( data, dico, x, y, z ) :
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros ( len ( z ) )
    j = 0
    for i in z:
        offset_z[j] = size_z       
        size_z *= len ( dico[i] )
        j += 1

    # création du tableau de contingence
    res = np.zeros ( size_z, dtype = object )

    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis ( lambda val_z : val_z.dot ( offset_z ),
                                         1, data[z,:].T )
        i = 0
        while i < size_z:
            indices, = np.where ( z_values == i )
            a,b,c = np.histogram2d ( data[x,indices], data[y,indices], 
                                     bins = [ len ( dico[x] ), len (dico[y] ) ] )
            res[i] = ( indices.size, a )
            i += 1
    else:
        a,b,c = np.histogram2d ( data[x,:], data[y,:], 
                                 bins = [ len ( dico[x] ), len (dico[y] ) ] )
        res[0] = ( data.shape[1], a )
    return res

def loi_du_khi_carre (Nxyz, Nxy, Nxz, Nyz, Nz) :
    return ( ( Nxyz - ( Nxz * Nyz / Nz ) ) ** 2 ) * Nz / ( Nxz * Nyz )

def sufficient_statistics_table ( contingency_table, x, y, z ):
    Nxyz  = np.array([ tab[1] for tab in contingency_table ])
    Nz    = np.array([ tab[0] for tab in contingency_table ])
    
    Nxy   = np.zeros(np.shape(Nxyz[0]))
    for xy in Nxyz:
        Nxy += xy
        
    Nxz = np.zeros((len(Nxyz), len(Nxyz[0])))
    for i, xy in enumerate(Nxyz):
        for xy_y in xy.T:
            Nxz[i] += xy_y
            
    Nyz = np.zeros((len(Nxyz), len(Nxyz[0].T)))
    for i, xy in enumerate(Nxyz):
        for xy_x in xy:
            Nyz[i] += xy_x

    X = len(Nxy)
    Y = len(Nxy.T)
    Z = len(Nz)

    return Nz, Nxy, Nxz, Nyz, Nxyz, X, Y, Z
    
def sufficient_statistics ( data, dico, x, y, z ) :
    
    khi_carre = 0
    cont = create_contingency_table ( data, dico, x, y, z )
    Nz, Nxy, Nxz, Nyz, Nxyz, X, Y, Z = sufficient_statistics_table ( cont, x, y, z )

    for zi in range( Z ) :
        for xi in range( X ):
            for yi in range( Y ):
                if Nz[zi] <> 0 and Nxz[zi,xi] <> 0 and Nyz[zi,yi] <> 0 :
                    khi_carre += loi_du_khi_carre ( Nxyz[zi][xi,yi], Nxy[xi,yi], Nxz[zi,xi], Nyz[zi,yi], Nz[zi] )

    return ( khi_carre, ( X - 1 ) * ( Y - 1 ) * ( len([ i for i in Nz if i <> 0 ]) ) )

def test_case_sufficient_statistics (names, data, dico, res_data, res_dico) :
    print 'sufficient_statistics ( data, dico, 1, 2, [3]): \n', sufficient_statistics ( data, dico, 1, 2, [3])
    print 'sufficient_statistics ( data, dico, 0, 1, [2]): \n', sufficient_statistics ( data, dico, 0, 1, [2])
    print 'sufficient_statistics ( data, dico, 0, 1, [2, 3, 4] ): \n', sufficient_statistics ( data, dico, 0, 1, [2, 3, 4] )
    print 'sufficient_statistics ( data, dico, 0, 3, [3, 4] ): \n', sufficient_statistics ( data, dico, 0, 3, [3, 4] )
    print 'sufficient_statistics ( data, dico, 1, 2, [3, 4] ): \n', sufficient_statistics ( data, dico, 1, 2, [3, 4])

def D_min (X, Y, Z) :
    return 5 * X * Y * Z
    
def indep_score( data, dico, x, y, z ) :
    
    cont = create_contingency_table ( data, dico, x, y, z )
    Nz, Nxy, Nxz, Nyz, Nxyz, X, Y, Z = sufficient_statistics_table ( cont, x, y, z )
    (khi_carre, DoF) = sufficient_statistics ( data, dico, x, y, z )
    
    if D_min (X, Y, Z) < len(data) :
        # ( khi_carre, DoF ) = ( -1, 1 )
        return ( -1, 1 )

    return stats.chi2.sf ( khi_carre, DoF ), DoF

def test_case_indep_score (names, data, dico, res_data, res_dico) :
    print 'indep_score( data, dico, 1, 3, [] ): \n', indep_score( data, dico, 1, 3, [] )
    print 'indep_score( data, dico, 1, 7, [] ): \n', indep_score( data, dico, 1, 7, [] )
    print 'indep_score( data, dico, 0, 1, [2, 3] ): \n', indep_score( data, dico, 0, 1, [2, 3] )
    print 'indep_score( data, dico, 1, 2, [3, 4] ): \n', indep_score( data, dico, 1, 2, [3, 4] )
    
def best_candidate ( data, dico, x, z, alpha ) :

    p_valeurs = []
    
    for y in range( x ):
        p_val, DoF = indep_score( data, dico, x, y, z )
        p_valeurs.append(p_val)

    if len(p_valeurs) == 0 or min( p_valeurs ) > alpha :
            return []
            
    return [ np.argmin(p_valeurs) ]

def test_case_best_candidate (names, data, dico, res_data, res_dico) :
    print 'best_candidate ( data, dico, 1, [], 0.05 ): \n', best_candidate ( data, dico, 1, [], 0.05 )
    print 'best_candidate ( data, dico, 4, [], 0.05 ): \n', best_candidate ( data, dico, 4, [], 0.05 )
    print 'best_candidate ( data, dico, 4, [1], 0.05 ): \n', best_candidate ( data, dico, 4, [1], 0.05 )
    print 'best_candidate ( data, dico, 5, [], 0.05 ): \n', best_candidate ( data, dico, 5, [], 0.05 )
    print 'best_candidate ( data, dico, 5, [6], 0.05 ): \n', best_candidate ( data, dico, 5, [6], 0.05 )
    print 'best_candidate ( data, dico, 5, [6,7], 0.05 ): \n', best_candidate ( data, dico, 5, [6,7], 0.05 )
    
def parents_merge(dico, list_tgt, list_src):

    count = 0
    tab   = np.zeros(len(dico))
    
    for i in list_tgt:
        tab[i] = 1
        
    for i in list_src:
        if tab[i] <> 1:
            tab[i] = 1
            count += 1

    return [ i for i, value in enumerate(tab) if value == 1 ], count
            
def create_parents ( data, dico, x, alpha ):
    parents = []
    for i in range(0, x):
        parents, add_num = parents_merge( dico, parents, best_candidate ( data, dico, x, parents, alpha ) )
            
    return parents

def test_case_create_parents (names, data, dico, res_data, res_dico) :
    
    print 'create_parents ( data, dico, 1, 0.05 ):\n', create_parents ( data, dico, 1, 0.05 )
    print 'create_parents ( data, dico, 4, 0.05 ):\n', create_parents ( data, dico, 4, 0.05 )
    print 'create_parents ( data, dico, 5, 0.05 ):\n', create_parents ( data, dico, 5, 0.05 )
    print 'create_parents ( data, dico, 6, 0.05 ):\n', create_parents ( data, dico, 6, 0.05 )
    
def learn_BN_structure ( data, dico, alpha ) :
    return [ create_parents ( data, dico, x, alpha ) for x in range(len(dico)) ]
    
def display_BN ( node_names, bn_struct, bn_name, style ):
    graph = pydot.Dot( bn_name, graph_type='digraph')

    # création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node( name, 
                               style="filled",
                               fillcolor=style["bgcolor"],
                               fontcolor=style["fgcolor"] )
        graph.add_node( new_node )

    # création des arcs
    for node in range ( len ( node_names ) ):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge ( node_names[par], node_names[node] )
            graph.add_edge ( new_edge )

    # sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png( outfile )
    img = mpimg.imread ( outfile )
    plt.imshow( img )

def learn_parameters ( bn_struct, ficname ):
    # création du dag correspondant au bn_struct
    graphe = gum.DAG ()
    nodes = [ graphe.addNode () for i in range ( len(bn_struct) ) ]
    for i in range ( len(bn_struct) ):
        for parent in bn_struct[i]:
            graphe.addArc ( nodes[parent], nodes[i] )

    # appel au BNLearner pour apprendre les paramètres
    learner = gum.BNLearner ( ficname )
    learner.useScoreLog2Likelihood ()
    learner.useAprioriSmoothing ()
    return learner.learnParameters ( graphe )

def test_case_learn_BN_structure (names, data, dico, res_data, res_dico, img_name) :
    style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }
    bn_struct = learn_BN_structure ( data, dico, 0.05 )
    print 'learn_BN_structure ( data, dico, 0.05 ): \n', bn_struct
    display_BN ( names, bn_struct, img_name, style )

def apprentissage_calcul_probabiliste (names, data, dico, res_data, res_dico, ficname) :
    # création du réseau bayésien à la aGrUM
    bn_struct = learn_BN_structure ( data, dico, 0.05 )
    bn = learn_parameters ( bn_struct, ficname )

    # affichage de sa taille
    print bn
    
    # récupération de la ''conditional probability table'' (CPT) et affichage de cette table
    pretty_cpt ( bn.cpt ( bn.idFromName ( 'bronchitis?' ) ) )
    
    gnb.showPosterior ( bn, {}, 'bronchitis?' )
    gnb.showPosterior ( bn, {'smoking?': 'true', 'tuberculosis?' : 'false' }, 'bronchitis?' )
    
def calcul_probabiliste (names, data, dico, res_data, res_dico, ficname) :
    # création du réseau bayésien à la aGrUM
    bn_struct = learn_BN_structure ( data, dico, 0.05 )
    bn = learn_parameters ( bn_struct, ficname )
    
    print bn

    # pretty_cpt(bn.cpt(bn.idFromName('gill spacing')))
    
    # ---- P ( class = p ), p signifie que votre champignon est vénéneux ----
    proba = gnb.getPosterior ( bn, {}, 'class' )
    pretty_cpt ( proba )

    # ---- P ( gill spacing | class = p ) ----
    proba = gnb.getPosterior ( bn, {'class':'p'}, 'gill spacing' )
    pretty_cpt ( proba )
    # gnb.showPosterior ( bn, {'class':'p'}, 'gill spacing' )
    # pretty_cpt ( bn.cpt ( bn.idFromName ( 'gill spacing' ) ) )

    # ---- P ( class | ring_number = o, gill size = n, cap shape = b ) ----
    proba = gnb.getPosterior ( bn, { 'ring number' : 'o', 'gill size' : 'n', 'cap shape' : 'b' }, 'class' )
    pretty_cpt ( proba )
    # gnb.showPosterior ( bn, { 'ring number' : 'o', 'gill size' : 'n', 'cap shape' : 'b' }, 'class' )
    # pretty_cpt ( bn.cpt ( bn.idFromName ( 'class' ) ) )
        
def main():
    fname = '2014_tme5_agaricus_lepiota.csv'
    names, data, dico = read_csv ( fname )
    ( res_data, res_dico ) = translate_data ( data )

    # apprentissage_calcul_probabiliste (names, data, dico, res_data, res_dico, fname)
    
    # ---- ---- Approfondissements sur le TME 5 ---- ----

    # -- 1. Structure du réseau agaricus-lepiota -- 
    test_case_learn_BN_structure (names, data, dico, res_data, res_dico, 'AgaricusLepiota')

    # -- 2. Calculs de probabilité --
    calcul_probabiliste (names, data, dico, res_data, res_dico, fname)
    
if __name__ == "__main__":
    main()
