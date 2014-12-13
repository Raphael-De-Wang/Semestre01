#! /usr/bin/python

# -*- coding: utf-8 -*-

##### Projet MOGPL #####

# Algorithmes pour le partage equitable de biens indivisibles

# Jinxin WANG 3404759
# Yvan SRAKA 3401082

import math
import time
import numpy as np
import gurobipy as gb

from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow

# classe abstraite
class Modelisation(object):
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = gb.Model(self.model_name)
        self.var_tb = []
        self.expr = gb.LinExpr()
        self.model_initialization()

    def valeurs_table_aleatoire(self, n, m, M):
        return np.round(np.random.triangular(0, M / 2, M, (n, m)))

    def model_initialization(self):
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
        self.model.write(self.model_name + ".ilp")

    def get_solution_optimal(self):
        return self.model.objVal

    def get_runtime(self):
        return self.model.Runtime
    
    def rapport(self, fname, contenu):
        fb = open(fname,'a')
        fb.write(contenu)
        fb.close()

class Simple_Model(Modelisation):

    def __init__(self, n, m, M, model_name):
        self.agent_num = n
        self.objet_num = m
        # self.satisfaction_tb = self.valeurs_table_aleatoire(n, m, M)
        self.satisfaction_tb = np.array([[3,6,17,6,12],[6,2,12,13,5],[10,10,4,7,13],[14,17,10,6,19],[11,12,9,10,6]])
        Modelisation.__init__(self, model_name)

    def declarer_variables(self):
        self.var_tb = [ [ self.model.addVar(vtype=gb.GRB.BINARY, name="x(%d,%d)"%(i,j)) for j in range(self.objet_num) ] for i in range(self.agent_num) ]
        self.model.update()

    def definir_contraintes(self):
        for i in range(self.agent_num):
            self.model.addConstr(gb.quicksum([self.var_tb[i][j] for j in range(self.objet_num) ]) <= 1, "Contrainte-AO%d"%(i))
        
        for j in range(self.objet_num):
            self.model.addConstr(gb.quicksum([self.var_tb[i][j] for i in range(self.agent_num) ]) <= 1, "Contrainte-OA%d"%(j))

    def definir_objectif(self):
        for i in range(self.agent_num):
            for j in range(self.objet_num):
                self.expr += self.satisfaction_tb[i,j] * self.var_tb[i][j]
        self.model.setObjective(self.expr, gb.GRB.MAXIMIZE)

    def solution(self):
        obj_lst = []
        for agent in range(self.agent_num):
            for objet in range(self.objet_num):
                if self.var_tb[agent][objet].x == 1:
                    valeur = self.satisfaction_tb[agent,objet]
                    obj_lst.append([objet, valeur])
                    break
        return np.array(obj_lst)

class Approche_Egalitariste_MaxMin(Simple_Model):

    def declarer_variables(self):
        super(Approche_Egalitariste_MaxMin,self).declarer_variables()
        self.z = self.model.addVar(vtype=gb.GRB.INTEGER, name="z")
        self.model.update()

    def definir_contraintes(self):
        super(Approche_Egalitariste_MaxMin,self).definir_contraintes()
        for i in range(self.agent_num):
            self.model.addConstr(self.z <= gb.quicksum([ self.var_tb[i][j] * self.satisfaction_tb[i,j] for j in range(self.objet_num)]), "Contrainte-z%d"%(i))

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(self.z), gb.GRB.MAXIMIZE)

class Approche_Egalitariste_MaxMean(Simple_Model):

    def declarer_variables(self):
        super(Approche_Egalitariste_MaxMin,self).declarer_variables()
        self.avg = self.model.addVar(vtype=gb.GRB.CONTINUOUS, name="avg")
        self.model.update()

    def definir_contraintes(self):
        super(Approche_Egalitariste_MaxMin,self).definir_contraintes()
        self.model.addConstr(self.avg <= gb.quicksum([self.var_tb[i][j] * self.satisfaction_tb[i,j] for i in range(self.agent_num) for j in range(self.objet_num)]) / self.agent_num, \
                             "Contrainte(%d,%d)"%(i,j))

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(self.avg), gb.GRB.MAXIMIZE)

class Approche_Egalitariste_MaxMin_Epsilon(Approche_Egalitariste_MaxMin):

    def __init__(self, n, m, M, model_name):
        self.epsilon = 0.01
        Approche_Egalitariste_MaxMin.__init__(self, n, m, M, model_name)

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(
            self.z + self.epsilon * gb.quicksum([ self.var_tb[i][j] * self.satisfaction_tb[i,j] for i in range(self.agent_num) for j in range(self.objet_num) ])), gb.GRB.MAXIMIZE)

class Approche_Egalitariste_MinRegrets(Simple_Model):

    def declarer_variables(self):
        super(Approche_Egalitariste_MinRegrets,self).declarer_variables()
        self.regrets = self.model.addVar(vtype=gb.GRB.INTEGER, name="regrets")
        self.model.update()

    def definir_contraintes(self):
        super(Approche_Egalitariste_MinRegrets,self).definir_contraintes()
        for i in range(self.agent_num):
            self.model.addConstr(self.regrets >= max(self.satisfaction_tb[i]) - gb.quicksum([ self.var_tb[i][j] * self.satisfaction_tb[i,j] for j in range(self.objet_num)]), \
                                 "Contrainte-regret%d"%(i))

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(self.regrets), gb.GRB.MINIMIZE)

#### Bonus #####
class OWA_MaxKAgents(Modelisation):
    def __init__(self, k, model_name, solution_fixe):
        self.k = k
        self.solv = solution_fixe
        Modelisation.__init__(self, model_name)
        
    def declarer_variables(self):
        self.var_tb = [ self.model.addVar(vtype=gb.GRB.CONTINUOUS, name="y(%d)"%(i)) for i in range(self.k) ]
        self.model.update()

    def definir_contraintes(self):
        self.model.addConstr(gb.quicksum([ self.var_tb[i] for i in range(self.k) ]) >= self.k, "Contrainte-sigma-y")
        for i,y in enumerate(self.var_tb):
            self.model.addConstr( y <= 1, "Contrainte-sb-y%d"%(i))
            self.model.addConstr( y >= 0, "Contrainte-ib-y%d"%(i))
            
    def definir_objectif(self):
        self.model.setObjective(gb.quicksum([ self.var_tb[i] * self.solv[i] for i in range(self.k) ]), gb.GRB.MINIMIZE)

    def solution(self):
        return np.array([ self.var_tb[i].x for i in range(self.k)])
        
def ex3():
    N        = [5]
    M        = [20]
    num_iter = 1
    fname    = "MOGPL_REPORT_Ex03.txt"
    general_test(N,M,num_iter,fname,Simple_Model)

def ex6():
    N        = [5]# [10,50,100]
    M        = [20]# [100]
    num_iter = 1
    fname    = "MOGPL_REPORT_Ex06.txt"
    general_test(N,M,num_iter,fname,Approche_Egalitariste_MaxMin)

def ex8():
    N        = [10,50,100]
    M        = [100]
    num_iter = 10
    fname    = "MOGPL_REPORT_Ex08.txt"
    general_test(N,M,num_iter,fname,Approche_Egalitariste_MaxMin_Epsilon)
    
def ex11():
    N        = [10,50,100]
    M        = [100]
    num_iter = 10
    fname    = "MOGPL_REPORT_Ex11.txt"
    general_test(N,M,num_iter,fname,Approche_Egalitariste_MinRegrets)

def ex13():
    k = 3
    n = 10
    m = 100
    model_name = "MaxKAgents"
    md = Simple_Model(n, n, m, "model-simple")
    md.resoudre()
    solv_fixe = np.sort(md.solution()[:,1])
    md = OWA_MaxKAgents(k, model_name, solv_fixe)
    md.resoudre()
    print md.solution()
    
def general_test( N, M, num_iter, fname, Model):
    
    report = "Resultat: \n"
    
    for m in M:
        report += "\n\n[ M = %d ]\n"%(m)
        report += "---------------------------------------------------------------\n"
        report += "\tn\tt\tmoy/M\tmin/M\tmax/M\n"
        report += "---------------------------------------------------------------\n"
        for n in N:
            valeur_table = []
            t1 = time.time()
            for i in range(num_iter):
                md = Model(n, n, m, "model_m_eq_n_%d%d%d"%(m,n,i))
                md.resoudre()
                obj_list = md.solution()
                valeur_table.append(obj_list[:,1])
                '''
                for agent, ov in enumerate(obj_list):
                    print "Agent %d prend Objet: %d Valeur: %d"%(agent,ov[0],ov[1])
                '''
            t2 = time.time()
            moy_M = 1. * np.sum(valeur_table) / n / num_iter / m
            min_M = 1. * np.min(valeur_table) / m
            max_M = 1. * np.max(valeur_table) / m
            t = np.round( t2 - t1 )
            report += "\t%d\t%d\t%.2f\t%.2f\t%.2f\n"%(n,t,moy_M,min_M,max_M)

    md.rapport(fname, report)
    
if __name__ == '__main__':
    ex13()
    ex6()
