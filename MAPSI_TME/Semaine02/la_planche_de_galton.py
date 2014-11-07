import numpy as np
import matplotlib.pyplot as plt

def loi_bernoulli(p):
    res = np.random.rand(1)[0]
    if p >= res:
        return 1
    return 0

def loi_binomiale(n, p):
    res = np.zeros(n)
    for i in range(n):
        res[i] = loi_bernoulli(p)
    return res.sum();

def tableau_cases(inst, n=13, p=0.5):
    cases = np.zeros(n)
    for i in range(inst):
        cases[loi_binomiale(n, p)-1] += 1
    return cases

def repeat_binomiale(inst, n, p):
    res = np.zeros(inst)
    for i in range(inst):
        res[i] = loi_binomiale(n, p)
    return res
    
def hist_binomiale(tableau_1000_cases, nb_bins):
    plt.hist(tableau_1000_cases, nb_bins,(0,nb_bins-1))
    plt.show()
    
def testcase():
    n = 20
    p = 0.5
    inst = 1000
    print loi_bernoulli(p)
    print loi_binomiale(n,p)
    # tableau = tableau_cases(inst)
    tableau = repeat_binomiale(inst, n, p)
    print tableau
    hist_binomiale(tableau, n)
    
    
def main():
    testcase()

if __name__ == "__main__":
    main()
