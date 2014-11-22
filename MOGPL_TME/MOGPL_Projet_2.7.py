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


def get_random_table_values(n, m, M):
    return np.round(np.random.triangular(0, M / 2, M, (n, m)))


class Modeling:

    def __init__(self, constr_a, constr_b, coe_func_obj, model_name):
        if len(np.shape(constr_a)) == 1:
            self.constr_a = [constr_a]
        else:
            self.constr_a = constr_a
        self.constr_b = constr_b
        self.coe_func_obj = coe_func_obj
        self.model_name = model_name
        self.model = gb.Model(self.model_name)
        self.nbconst = len(self.constr_a)
        self.nbvar = len(self.coe_func_obj)
        self.var_list = []
        self.expr = gb.LinExpr()
        self.declare_variables()
        self.define_objective()
        self.define_constraints()

    def declare_variables(self):
        raise Exception(NotImplemented)

    def define_constraints(self):
        raise Exception(NotImplemented)

    def define_objective(self):
        raise Exception(NotImplemented)

    def solve(self):
        self.model.optimize()

    def save_model(self):
        self.model.write(self.model_name + ".ilp")

    def get_optimal_solution(self):
        return self.model.objVal

    def get_runtime(self):
        return self.model.Runtime

    def report(self, fname, contenu):
        fb = open(fname, 'a')
        fb.write(contenu)
        fb.close()


class Simple_Model(Modeling):

    def __init__(self, n, m, M, model_name):
        self.val_tb = get_random_table_values(n, m, M)
        self.val_tb_shape = (n, m)
        constr_a = np.zeros((n + m, n * m))
        for i in range(n):
            constr_a[i, i * m:(i + 1) * m] = 1
            for j in range(m):
                constr_a[n + j, i * m + j] = 1
        constr_b = np.ones(n + m)
        coe_func_obj = [val for i, ligne in enumerate(self.val_tb) for j, val in enumerate(ligne)]
        Modeling.__init__(self, constr_a, constr_b, coe_func_obj, model_name)

    def agent_num(self):
        return self.val_tb_shape[0]

    def object_num(self):
        return self.val_tb_shape[1]

    def optimal_solution_formula(self):
        return gb.quicksum([self.constr_a[i][j] * self.var_list[j] * self.val_tb[i][j % self.object_num()] for i in range(self.nbconst) for j in range(self.nbvar) if i < self.agent_num()])

    def agent_optimal_solution_formula(self, agent):
        if agent >= self.agent_num():
            raise Exception("Erreur: L'agent %d n'existe pas." % agent)
        return gb.quicksum([self.constr_a[agent][j] * self.var_list[j] * self.val_tb[agent][j % self.object_num()] for j in range(self.nbvar)])

    def declare_variables(self):
        self.var_list = [self.model.addVar(vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.model.update()

    def define_constraints(self):
        for i in range(self.nbconst):
            self.model.addConstr(gb.quicksum([self.constr_a[i][j] * self.var_list[j] for j in range(self.nbvar)]) <= self.constr_b[i], "Contrainte%d" % (i))

    def define_objective(self):
        for i in range(self.nbvar):
            self.expr += self.coe_func_obj[i] * self.var_list[i]
        self.model.setObjective(self.expr, gb.GRB.MAXIMIZE)

    def agent_takes_object(self):
        obj_lst = []
        for j in range(self.nbvar):
            if self.var_list[j].x == 1:
                # agent = j / self.object_num()
                object_ = j % self.object_num()
                # print "L'agent %d prend l'object %d\tvalue: %d\tvalue max:
                # %d" % (agent, object, self.val_tb[agent][object],
                # max(self.val_tb[agent]))
                obj_lst.append(object)
        return obj_lst

    def agent_takes_value(self):
        val_lst = []
        for j in range(self.nbvar):
            if self.var_list[j].x == 1:
                agent = j / self.object_num()
                object_ = j % self.object_num()
                value = self.val_tb[agent][object]
                val_lst.append(value)
        return val_lst


class Egalitarian_Approach_MaxMin(Simple_Model):

    def declare_variables(self):
        self.var_list = [self.model.addVar(vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.z = self.model.addVar(vtype=gb.GRB.INTEGER, name="z")
        self.model.update()

    def define_constraints(self):
        for i in range(self.nbconst):
            self.model.addConstr(gb.quicksum([self.constr_a[i][j] * self.var_list[j] for j in range(self.nbvar)]) <= self.constr_b[i], "Contrainte%d" % (i))
            if i < self.agent_num():
                self.model.addConstr(self.z <= gb.quicksum([self.constr_a[i][j] * self.var_list[j] * self.val_tb[i][j % self.object_num()] for j in range(self.nbvar)]), "Contrainte%d.2" % (i))

    def define_objective(self):
        self.model.setObjective(gb.LinExpr(self.z), gb.GRB.MAXIMIZE)


class Egalitarian_Approach_MaxMean(Simple_Model):

    def declare_variables(self):
        self.var_list = [self.model.addVar(vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.avg = self.model.addVar(vtype=gb.GRB.CONTINUOUS, name="avg")
        self.model.update()

    def define_constraints(self):
        for i in range(self.nbconst):
            self.model.addConstr(gb.quicksum([self.constr_a[i][j] * self.var_list[j] for j in range(self.nbvar)]) <= self.constr_b[i], "Contrainte%d" % (i))
        self.model.addConstr(self.avg <= self.optimal_solution_formula() / self.agent_num(), "Contrainte%d" % (i + 1))

    def define_objective(self):
        self.model.setObjective(gb.LinExpr(self.avg), gb.GRB.MAXIMIZE)


class Egalitarian_Approach_MaxMin_Epsilon(Egalitarian_Approach_MaxMin):

    def __init__(self, n, m, M, model_name):
        self.epsilon = 0.01
        Egalitarian_Approach_MaxMin.__init__(self, n, m, M, model_name)

    def define_objective(self):
        self.model.setObjective(gb.LinExpr(self.z + self.epsilon * self.optimal_solution_formula()), gb.GRB.MAXIMIZE)


class Egalitarian_Approach_MinRegrets(Simple_Model):

    def declare_variables(self):
        self.var_list = [self.model.addVar(vtype=gb.GRB.BINARY, name="x%d" % i) for i in range(self.nbvar)]
        self.regrets = self.model.addVar(vtype=gb.GRB.INTEGER, name="regrets")
        self.model.update()

    def define_constraints(self):
        for i in range(self.nbconst):
            self.model.addConstr(gb.quicksum([self.constr_a[i][j] * self.var_list[j] for j in range(self.nbvar)]) <= self.constr_b[i], "Contrainte%d" % (i))
            if i < self.agent_num():
                self.model.addConstr(self.regrets >= max(
                    self.val_tb[i]) - self.agent_optimal_solution_formula(i), "Contrainte%d.2" % (i))

    def define_objective(self):
        self.model.setObjective(gb.LinExpr(self.regrets), gb.GRB.MINIMIZE)


class Egalitarian_Approach_MaxFlow:

    def __init__(self, n, m, M, model_name):
        self.n = n
        self.m = m
        self.M = M
        self.model_name = model_name
        self.solution = []
        self.graph = digraph()

    def max_flow_with_lambda(self, l):
        val_tb = get_random_table_values(n, m, M)
        self.graph.add_nodes(["s", "p"])
        for j in range(m):
            self.graph.add_node("o%s" % j)
            self.graph.add_edge(("o%s" % j, "p"), 1)
        for i in range(n):
            self.graph.add_node("a%s" % i)
            self.graph.add_edge(("s", "a%s" % i), 1)
            for j in range(m):
                if val_tb[i][j] >= l:
                    self.graph.add_edge(("a%s" % i, "o%s" % j), 1)
        return maximum_flow(gr, "s", "p")

    def solve(self):
        _lambda = self.M
        is_instance = False
        while not is_instance:
            _max_flow = self.max_flow_with_lambda(_lambda)
            is_instance = True
            for i in range(self.n):
                if not _max_flow[1]["a%s" % i]:
                    is_instance = False
            _lambda -= 1
        self.solution = [i for i in _max_flow[0] if _max_flow[
            0][i] and not "s" in i and not "p" in i], _lambda

    def report(self):
        for i in self.solution:
            print "L'agent %d prend l'objet %d" % i

    def save(self):
        pygraph.to_file(self.graph, 'graph.dat')

    def show(self):
        from pygraphlib import pydot
        dot = pydot.Dot(self.graph)
        dot.save_image('graph_1.gif', mode='gif')
        for edge in self.graph.edge_list():
            head, tail = self.graph.edge_nodes(edge)
            dot.set_edge_style(head, tail, label=self.graph.get_edge_wt(edge))
        dot.save_image('graph_2.gif', mode='gif')


def output_test_results(N, M, num_iter, fname, Model):
    report = "Resultat: \n"
    for m in M:
        for n in N:
            value_table = []
            report += "\n\n[ M = %d ]\n" % (m)
            report += "---------------------------------------------------------------\n"
            report += "\tn\tt\tmoy/M\tmin/M\tmax/M\n"
            report += "---------------------------------------------------------------\n"
            t1 = time.time()
            for i in range(num_iter):
                md = Model(n, n, m, "model_m_eq_n_%d%d%d" % (m, n, i))
                md.solve()
                value_table.append(md.agent_takes_value())
            t2 = time.time()
            moy_M = 1. * np.sum(value_table) / n / num_iter / m
            min_M = 1. * np.min(value_table) / m
            max_M = 1. * np.max(value_table) / m
            t = np.round(t2 - t1)
            report += "\t%d\t%d\t%.2f\t%.2f\t%.2f" % (n, t, moy_M, min_M, max_M)
    md.report(fname, report)


if __name__ == '__main__':
    # --- Ex 3 ---
    N = [10, 50, 100] # , 500, 1000]
    M = [10, 100, 1000]
    num_iter = 10
    fname = "MOGPL_REPORT_Ex03.txt"
    output_test_results(N, M, num_iter, fname, Simple_Model)
    # --- Ex 6 ---
    N = [10, 50, 100]
    M = [100]
    num_iter = 10
    fname = "MOGPL_REPORT_Ex06.txt"
    output_test_results(N, M, num_iter, fname, Egalitarian_Approach_MaxMin)
    # --- Ex 11 ---
    N = [10, 50, 100]
    M = [100]
    num_iter = 10
    fname = "MOGPL_REPORT_Ex11.txt"
    output_test_results(N, M, num_iter, fname, Egalitarian_Approach_MinRegrets)
