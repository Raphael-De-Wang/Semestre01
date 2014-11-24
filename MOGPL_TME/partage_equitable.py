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

def valeurs_table_aleatoire(n, m, M):
    return np.round(np.random.triangular(0, M / 2, M, (n, m)))

# classe abstraite
class Modelisation:
    
    def __init__(self, coe_func_obj, model_name):
        self.coe_func_obj = coe_func_obj
        self.model_name = model_name
        self.model = gb.Model(self.model_name)
        self.nbvar = len(self.coe_func_obj)
        self.var_list = []
        self.expr = gb.LinExpr()

        self.declarer_variables()
        self.definir_objectif()
        self.definir_contraintes()

    def contr_a(self, ligne, colone):
        raise Exception(NotImplemented)

    def contr_b(self, i):
        raise Exception(NotImplemented)
        
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
        self.val_tb = valeurs_table_aleatoire(n, m, M)
        self.val_tb_shape = (n, m)
        self.nbcont = n + m
        coe_func_obj = [val for i, ligne in enumerate(self.val_tb) for j, val in enumerate(ligne)]
        Modelisation.__init__(self, coe_func_obj, model_name)

    def contr_a(self, ligne, colone):
        (n, m) = self.val_tb_shape
        if 0 <= ligne and ligne < n:
            if colone >= ligne * m and colone < (ligne + 1) * m :
                return 1
        elif 0 <= ligne and ligne < m + n:
            if ( colone - ( ligne - n ) ) % m == 0:
                return 1
        return 0
            
    def contr_b(self, i):
        if 0 <= i and i < self.nbcont:
            return 1
        
    def agent_num(self):
        return self.val_tb_shape[0]

    def objet_num(self):
        return self.val_tb_shape[1]

    def formule_solution_optimale(self):
        return gb.quicksum([self.contr_a(i,j) * self.var_list[j] * self.val_tb[i][j % self.objet_num()] for i in range(self.nbcont) for j in range(self.nbvar) if i < self.agent_num()])

    def formule_agent_solution_optimale(self, agent):
        if agent >= self.agent_num():
            raise Exception("Erreur: L'agent %d n'existe pas." % agent)

        return gb.quicksum([self.contr_a(agent,j) * self.var_list[j] * self.val_tb[agent][j % self.objet_num()] for j in range(self.nbvar)])

    def declarer_variables(self):
        self.var_list = [self.model.addVar(
            vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.model.update()

    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr(gb.quicksum([self.contr_a(i,j) * self.var_list[j] for j in range(self.nbvar)]) <= self.contr_b(i), "Contrainte%d" % (i))

    def definir_objectif(self):
        for i in range(self.nbvar):
            self.expr += self.coe_func_obj[i] * self.var_list[i]
        self.model.setObjective(self.expr, gb.GRB.MAXIMIZE)

    def agent_prend_objet(self):
        obj_lst = []
        for j in range(self.nbvar):
            if self.var_list[j].x == 1:
                agent = j / self.objet_num()
                objet = j % self.objet_num()
                print "L'agent %d prend l'objet %d\tvaleur: %d\tvaleur max: %d" % (agent, objet, self.val_tb[agent][objet], max(self.val_tb[agent]))
                obj_lst.append(objet)
        return obj_lst

    def agent_prend_valeur(self):
        val_lst = []
        for j in range(self.nbvar):
            if self.var_list[j].x == 1:
                agent = j / self.objet_num()
                objet = j % self.objet_num()
                valeur = self.val_tb[agent][objet]
                val_lst.append(valeur)
        return val_lst

class Approche_Egalitariste_MaxMin(Simple_Model):

    def declarer_variables(self):
        self.var_list = [self.model.addVar(
            vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.z = self.model.addVar(vtype=gb.GRB.INTEGER, name="z")
        self.model.update()

    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr(gb.quicksum([self.contr_a(i,j) * self.var_list[j] for j in range(self.nbvar)]) <= self.contr_b(i), "Contrainte%d" % (i))
            if i < self.agent_num():
                self.model.addConstr(self.z <= gb.quicksum([self.contr_a(i,j) * self.var_list[j] * self.val_tb[i][j % self.objet_num()] for j in range(self.nbvar)]), "Contrainte%d.2" % (i))

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(self.z), gb.GRB.MAXIMIZE)

class Approche_Egalitariste_MaxMean(Simple_Model):

    def declarer_variables(self):
        self.var_list = [self.model.addVar(
            vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.avg = self.model.addVar(vtype=gb.GRB.CONTINUOUS, name="avg")
        self.model.update()

    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr(gb.quicksum([self.contr_a(i,j) * self.var_list[j] for j in range(self.nbvar)]) <= self.contr_b(i), "Contrainte%d" % (i))

        self.model.addConstr(self.avg <= self.formule_solution_optimale() / self.agent_num(), "Contrainte%d" % (i + 1))

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(self.avg), gb.GRB.MAXIMIZE)

class Approche_Egalitariste_MaxMin_Epsilon(Approche_Egalitariste_MaxMin):

    def __init__(self, n, m, M, model_name):
        self.epsilon = 0.01
        Approche_Egalitariste_MaxMin.__init__(self, n, m, M, model_name)

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(
            self.z + self.epsilon * self.formule_solution_optimale()), gb.GRB.MAXIMIZE)

class Approche_Egalitariste_MinRegrets(Simple_Model):

    def declarer_variables(self):
        self.var_list = [self.model.addVar(
            vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.regrets = self.model.addVar(vtype=gb.GRB.INTEGER, name="regrets")
        self.model.update()

    def definir_contraintes(self):
        for i in range(self.nbcont):
            self.model.addConstr(gb.quicksum([self.contr_a(i,j) * self.var_list[j] for j in range(self.nbvar)]) <= self.contr_b(i), "Contrainte%d" % (i))
            if i < self.agent_num():
                self.model.addConstr(self.regrets >= max(self.val_tb[i]) - self.formule_agent_solution_optimale(i), "Contrainte%d.2" % (i))

    def definir_objectif(self):
        self.model.setObjective(gb.LinExpr(self.regrets), gb.GRB.MINIMIZE)

class Approche_Egalitariste_Max_Flow:

    def __init__(self, n, m, M, model_name):
        self.n = n
        self.m = m
        self.M = M
        self.model_name = model_name
        self.solution = []

    def max_flow_with_lambda(self, l):
        val_tb = valeurs_table_aleatoire(n, m, M)
        gr = digraph()
        gr.add_nodes(["s", "p"])
        for j in range(m):
            gr.add_node("o%s" % j)
            gr.add_edge(("o%s" % j, "p"), 1)
        for i in range(n):
            gr.add_node("a%s" % i)
            gr.add_edge(("s", "a%s" % i), 1)
            for j in range(m):
                if val_tb[i][j] >= l:
                    gr.add_edge(("a%s" % i, "o%s" % j), 1)
        return maximum_flow(gr, "s", "p")

    def resoudre(self):
        _lambda = self.M
        is_instance = False
        while not is_instance:
            _max_flow = self.max_flow_with_lambda(_lambda)
            is_instance = True
            for i in range(self.n):
                if not _max_flow[1]["a%s" % i]:
                    is_instance = False
            _lambda -= 1
        self.solution = [i for i in _max_flow[0] if _max_flow[0][i] and not "s" in i and not "p" in i], _lambda

    def rapport(self):
        for i in self.solution:
            print "L'agent %d prend l'objet %d" % i

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
                t_md        = time.time()
                md = Model(n, n, m, "model_m_eq_n_%d%d%d"%(m,n,i))
                t_md       -= time.time()
                t_resoudre  = time.time()
                md.resoudre()
                t_resoudre -= time.time()
                valeur_table.append(md.agent_prend_valeur())

            t2 = time.time()
            moy_M = 1. * np.sum(valeur_table) / n / num_iter / m
            min_M = 1. * np.min(valeur_table) / m
            max_M = 1. * np.max(valeur_table) / m
            t = np.round( t2 - t1 )
            report += "\t%d\t%d\t%.2f\t%.2f\t%.2f"%(n,t,moy_M,min_M,max_M)
            
    md.rapport(fname, report)
    md.agent_prend_objet()
    
def ex3():
    N        = [10,50,100] # ,500,1000]
    M        = [10,100,1000]
    num_iter = 10
    fname    = "MOGPL_REPORT_Ex03.txt"
    general_test(N,M,num_iter,fname,Simple_Model)

def ex6():
    N        = [10,50,100]
    M        = [100]
    num_iter = 10
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

if __name__ == '__main__':
    ex8()

