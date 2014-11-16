# -*- coding: utf-8 -*-
#### Projet MOGPL ####
# Algorithmes pour le partage equitable de biens indivisibles
# Wang Jinxin 3404759
# Yvan Sraka

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
        
    def get_solution_optimal(self):
        return self.model.objVal

    def get_runtime(self):
        return self.model.Runtime

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
        Modelisation.__init__(self, contr_a, contr_b, coe_func_obj, model_name)
        
    def agent_num(self):
        return self.val_tb_shape[0]
        
    def objet_num(self):
        return self.val_tb_shape[1]
        
    def formule_solution_optimale(self):
        return gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] * self.val_tb[i][j%self.objet_num()]  \
                                   for i in range(self.nbcont)                                                \
                                       for j in range(self.nbvar)                                             \
                                           if i < self.agent_num() ] )
                
    def formule_agent_solution_optimale(self, agent):
        if agent >= self.agent_num():
            raise Exception("Agent %d N'Existe PAS"%agent)
            
        return gb.quicksum( [ self.contr_a[agent][j] * self.var_list[j] * self.val_tb[agent][j%self.objet_num()] for j in range(self.nbvar) ] )
                              
    def declarer_variables(self):
        self.var_list = [ self.model.addVar( vtype=gb.GRB.BINARY, name="x%d"%i ) for i in range(self.nbvar) ]
        self.model.update()
        
    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr( gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] for j in range(self.nbvar) ] ) <= self.contr_b[i], "Contrainte%d"%(i))

    def definir_objectif(self):
        for i in range(self.nbvar):
            self.expr += self.coe_func_obj[i] * self.var_list[i]
        self.model.setObjective( self.expr, gb.GRB.MAXIMIZE )
        
    def rapport(self):
        print ""                
        print 'Solution Optimale:'
        for j in range(self.nbvar):
            if self.var_list[j].x == 1:
                agent = j/self.objet_num()
                objet = j%self.objet_num()
                print "Agent %d prend objet %d\tvaleur: %d\tvaluer max: %d"%( agent, objet, self.val_tb[agent][objet], max(self.val_tb[agent]))
        print 'Valeur de la fonction objectif :', self.model.objVal

class Approche_Egalitariste_MaxMin(Simple_Model):
    def declarer_variables(self):
        self.var_list = [ self.model.addVar( vtype=gb.GRB.BINARY, name="x%d"%i ) for i in range(self.nbvar) ]
        self.z = self.model.addVar( vtype=gb.GRB.INTEGER, name="z" )
        self.model.update()
        
    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr( gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] for j in range(self.nbvar) ] ) <= self.contr_b[i], "Contrainte%d"%(i))
            if i < self.agent_num():
                self.model.addConstr( self.z <= gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] * self.val_tb[i][j%self.objet_num()] for j in range(self.nbvar) ] ), "Contrainte%d.2"%(i))
        
    def definir_objectif(self):
        self.model.setObjective( gb.LinExpr(self.z), gb.GRB.MAXIMIZE )

class Approche_Egalitariste_MaxMean(Simple_Model):
    def declarer_variables(self):
        self.var_list = [ self.model.addVar( vtype=gb.GRB.BINARY, name="x%d"%i ) for i in range(self.nbvar) ]
        self.avg = self.model.addVar( vtype=gb.GRB.CONTINUOUS, name="avg" )
        self.model.update()

    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr( gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] for j in range(self.nbvar) ] ) <= self.contr_b[i], "Contrainte%d"%(i))
            
        self.model.addConstr( self.avg <= self.formule_solution_optimale() / self.agent_num(), "Contrainte%d"%(i+1))
                                       
    def definir_objectif(self):
        self.model.setObjective( gb.LinExpr(self.avg), gb.GRB.MAXIMIZE )

class Approche_Egalitariste_MaxMin_Epsillon(Approche_Egalitariste_MaxMin):
    def __init__(self, n, m, M, model_name):
        self.epsilon = 0.01
        Approche_Egalitariste_MaxMin.__init__(self, n, m, M, model_name)
        
    def definir_objectif(self):
        self.model.setObjective( gb.LinExpr(self.z + self.epsilon * self.formule_solution_optimale() ), gb.GRB.MAXIMIZE )

class Approche_Egalitariste_MinRegrets(Simple_Model):
    def __init__(self, n, m, M, model_name):
        Simple_Model.__init__(self, n, m, M, model_name)

    def declarer_variables(self):
        self.var_list = [ self.model.addVar( vtype=gb.GRB.BINARY, name="x%d"%i ) for i in range(self.nbvar) ]
        self.regrets = self.model.addVar( vtype=gb.GRB.INTEGER, name="regrets" )
        self.model.update()
        
    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr( gb.quicksum( [ self.contr_a[i][j] * self.var_list[j] for j in range(self.nbvar) ] ) <= self.contr_b[i], "Contrainte%d"%(i))
            if i < self.agent_num():
                self.model.addConstr( self.regrets >= max(self.val_tb[i]) - self.formule_agent_solution_optimale(i), "Contrainte%d.2"%(i))
        
    def definir_objectif(self):
        self.model.setObjective( gb.LinExpr(self.regrets), gb.GRB.MINIMIZE )

        
'''
sim = Simple_Model( 5, 5, 10, 'model_m_eq_n')
sim.resoudre()
sim.rapport()
print '\n\n'
ai = Approche_Egalitariste_MaxMin(5, 5, 10, 'max_min')
ai.resoudre()
ai.rapport()
print '\n\n'
mm = Approche_Egalitariste_MaxMean(5, 5, 10, 'max_mean')
mm.resoudre()
mm.rapport()
print '\n\n'
mme = Approche_Egalitariste_MaxMin_Epsillon(5,5,10,'maxmin_epsillon')
mme.resoudre()
mme.rapport()
'''
mr = Approche_Egalitariste_MinRegrets(5,5,10,'MinRegrets')
mr.resoudre()
mr.rapport()
'''
def test_case():
    M_lt    = [ 10, 100, 1000 ]
    n_lt    = [ 10, 50, 100, 500, 1000 ]
    nb_tire = 10
    for M in M_lt:
        print '\nM: %d'%(M)
        for n in n_lt:
            t_lt    = []
            sol_lt  = []
            print "------------------------------------------------"
            print "n\tt\tmoy/M\tmin/M\tmax/M"
            print "------------------------------------------------"
            for i in range(nb_tire):
                sim = Simple_Model( n, n, M, 'model_m_eq_n')
                sim.resoudre()
                # sim.raport()
                sol_lt.append(sim.get_solution_optimal())
                t_lt.append(sim.get_runtime())
            print "%d\t%d\t%d\t%d\t%d"%(n, sum(t_lt), np.mean(sol_lt)/M, min(sol_lt)/M, max(sol_lt)/M)
            
test_case()
'''
