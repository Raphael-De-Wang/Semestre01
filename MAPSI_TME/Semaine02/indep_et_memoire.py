import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

def nb_vars (P):
    return len(bin(len(P)).replace('0b','')) - 1

def load_P_from_txt(path):
    P = np.loadtxt(path)
    '''
    for i in range(len(P)):
        if P[i] == 0:
            P[i] =
    '''
    return P

def save_P_to_txt(path, P):
    return np.savetxt(path, P)
    
def marginalisation (P, index):
    step = 2<<index
    vect_0 = np.zeros(0)
    vect_1 = np.zeros(0)
    for i in range ( 0, len(P), step):
        vect_0 = np.concatenate((vect_0, P[i:i+(step>>1)]))
        vect_1 = np.concatenate((vect_1, P[i+(step>>1):i+step]))
        
    return vect_0 + vect_1

def expanse1Var ( P, index ):
    vect = np.zeros(len(P)<<1)
    mask = int('1'*len(P),2)
    if index != 0:
        mask = mask - int('1'*index,2)
    mask = ~mask
    for i in range(len(vect)):
        l = (i >> (index + 1)) << index
        r = mask & i
        vect[i] = P[r+l]
    return vect
    
def project ( P, ind_to_remove ):
    v = P
    ind_to_remove.sort ()
    for i in range ( ind_to_remove.size - 1, -1, -1 ):
        v = project1Var ( v, ind_to_remove[i] )
    return v
    
def proba_conditionnelle (P, index):
    '''
    P(Xn|X1,X2,...Xn-1) = P(X1,X2,...Xn)/P(X1,X2...Xn-1)
    '''
    marg = marginalisation (P, index)
    base = expanse1Var ( marg, index)
    return np.divide (P, base)

def is_indep( P, index, epsilon):
    probs = proba_conditionnelle (P, index)
    length = len(probs)
    dep_lst = np.zeros(nb_vars(P))
    for i in range(nb_vars(P)):
        if i == index:
            dep_lst[i] = -1
            continue
        diffs = probs[:length>>1] - probs[length>>1:length]
        length = length >> 1
        if max(diffs) > epsilon:
            dep_lst[i] = False
        else:
            dep_lst[i] = True
    return dep_lst

def find_indep ( P, epsilon ):
    rst = np.zeros((nb_vars(P),nb_vars(P)))
    for i in range(nb_vars(P)):
        rst[:][i] = is_indep( P, i, epsilon)
    return rst
    
def testcase():
    P = [0.05, 0.1, 0.15, 0.2, 0.02, 0.18, 0.13, 0.17]
    index = 0
    epsilon = 0.000006
    print P
    print marginalisation (P, 0)
    print nb_vars (P)
    print proba_conditionnelle (P, index)
    print is_indep( P, index, epsilon)
    
def testcase_proba_cond():
    ifname = '2014_tme2_asia.txt'
    ofname = '2014_tme2_save.txt'
    P = load_P_from_txt(ifname)
    index = nb_vars(P) - 1
    epsilon = 0.000006
    rst = proba_conditionnelle (P, index)
    print find_indep ( P, epsilon )
    # save_P_to_txt(ofname, rst)
    
def main():
    # testcase()
    testcase_proba_cond()
    
if __name__ == "__main__":
    main()
