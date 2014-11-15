# -*- coding: utf-8 -*-
#### Projet MOGPL ####
# Algorithmes pour le partage equitable de biens indivisibles
# Wang Jinxin 3404759
# Yvan

import math
import numpy as np
import gurobipy as gb

def valeurs_table_aleatoir (n, m, M) :
    return np.round(np.random.triangular(0, M/2, M, (n,m)))
    
# classe abstraite
class Modelisation :
    def __init__(self, contr_a, contr_b, coe_func_obj, model_name):
        
        if len(np.shape(contr_a)) == 1 :
            self.contr_a = [ contr_a ]
        else :
            self.contr_a = contr_a
            
        self.contr_b      = contr_b
        self.coe_func_obj = coe_func_obj
        self.model_name   = model_name
        self.model        = gb.Model(self.model_name)
        self.nbcont       = len(self.contr_a)
        self.nbvar        = len(self.coe_func_obj)
        self.var_list     = []
        self.expr         = gb.LinExpr()

        self.declarer_variables()
        self.definir_objectif()
        self.definir_contraintes()

    def declarer_variables(self):
        raise Exception(NotImplemented)
        
    def definir_contraintes(self):
        raise Exception(NotImplemented)

    def definir_objectif(self):
        raise Exception(NotImplemented)
        
    def resoudre(self):
        self.model.optimize()

    def save_model(self):
        self.model.write(self.model_name+".ilp")

class Simple_Model (Modelisation) :
    def __init__(self, n, m, M, model_name):
        self.val_tb = valeurs_table_aleatoir (n, m, M)
        self.val_tb_shape = (n, m)
        contr_a = np.zeros((n+m, n*m))
        for i in range(n):
            contr_a[i, i*m:(i+1)*m] = 1
            for j in range(m):
                contr_a[n+j, i*m+j] = 1
        contr_b = np.ones(n+m)
        coe_func_obj = [ val for i, ligne in enumerate(self.val_tb) for j, val in enumerate(ligne) ]
        print coe_func_obj
        print self.val_tb
        print contr_a
        Modelisation.__init__(self, contr_a, contr_b, coe_func_obj, model_name)
        
    def declarer_variables(self):
        self.var_list = [ self.model.addVar( vtype=gb.GRB.BINARY, name="x%d"%i ) for i in range(self.nbvar) ]
        self.model.update()
        
    def definir_objectif(self):
        for i in range(self.nbvar):
            self.expr += self.coe_func_obj[i] * self.var_list[i]
        self.model.setObjective( self.expr, gb.GRB.MAXIMIZE )
        
    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr( gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] for j in range(self.nbvar) ] ) <= self.contr_b[i], "Contrainte%d"%(i))

    def raport(self):
        print ""                
        print 'Solution optimale:'
        '''
        for j in range(self.nbvar):
            print 'x%d'%(j+1), '=', self.var_list[j].x
        '''
        for j in range(self.nbvar):
            if self.var_list[j].x == 1:
                print "Agent %d prend objet %d"%(j/self.val_tb_shape[1], j%self.val_tb_shape[1])
        print ""
        print 'Valeur de la fonction objectif :', self.model.objVal

    def get_solution_optimal(self):
        return self.model.objVal

    def get_runtime(self):
        return self.model.Runtime

def test_case():
    M = 10
    n_list = [10, 50, 100, 500, 1000]
    nb_tire = 10
    for n in n_list:
        sim = Simple_Model( 10, 10, 10, 'model_m_eq_n')
        sim.resoudre()
        sim.raport()
        print sim.get_solution_optimal()
        print sim.get_runtime()
        exit(0)

test_case()
