import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle as pkl

import dist_de_prob as ddp
import velib

def trace_alt_hist(alt, pAlt):
    # plt.figure(1, figsize=(8,4))
    # plt.plot(x,y,label="",color="red",linewidth=2)
    # plt.plot(x,z,"",label="")
    # plt.xlabel("")
    # plt.ylabel("")
    # plt.title("")
    # plt.legend()

    print pAlt
    plt.bar((alt[1:]+alt[:-1])/2, pAlt, alt[1]-alt[0])
    plt.show()
    # plt.savefig('alt.jpg') 
    
def testcase(res):
    alt = res[1]
    intervalle = alt[1]-alt[0]
    pAlt = res[0]/res[0].sum()
    pAlt /= intervalle # ??? pourquoi ?
    trace_alt_hist(alt, pAlt)
    
def main():
    data = velib.load_velibs_info('dataVelib.pkl')
    velib.defiler_stations(data)
    velib.defiler_stations(data)
    velib.defiler_stations(data)
    
    res = ddp.get_mP_Al(data)
    # res = ddp.get_mP_Ar_hist(data)    
    testcase(res)

if __name__ == "__main__":
    main()
