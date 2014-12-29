import numpy as np
import pickle as pkl
import os.path as op
import matplotlib.pyplot as plt

def trace_comp(trainRes, testRes, N, xRange, label):
    fig = plt.figure()
    if len(trainRes) != len(testRes):
        raise("InvalidDataSet")
    
    plt.plot(xRange, trainRes, label='Train')
    plt.plot(xRange,  testRes,  label='Test')
    plt.legend()
    plt.savefig("comparation[N=%d]%s.png"%(N,label))
    plt.close(fig)

def main():
    for N in range(2,8):
        print "N = %d"%N
        if op.exists("Res[N=%d,K=(5,200)].npy"%(N)):
            [trainRes, testRes, trainVarRes, testVarRes] = np.load("Res[N=%d,K=(5,200)].npy"%(N))
            kRange = range(5,200)
            trace_comp(trainRes,    testRes,    N, kRange, "Mean")
            trace_comp(trainVarRes, testVarRes, N, kRange, "Variance")            
if __name__ == "__main__":
    main()
