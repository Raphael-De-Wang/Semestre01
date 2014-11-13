# -*- coding: utf-8 -*-
#### Projet MOGPL ####
# Algorithmes pour le partage equitable de biens indivisibles
# Wang Jinxin 3404759
# Yvan
# Moncef

import math
import numpy as np

def create_satification_table(num_agents, num_bien, scale):
    return np.round( np.random.rand(num_agents, num_bien) * scale )

def create_exemplaires_table(num_bien):
    return np.round( np.random.rand(num_bien) * scale )
    
def contrante_matrice():
    
