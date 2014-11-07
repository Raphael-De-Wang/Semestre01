import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl

import velib

def get_mP_Ar(sttns_info):
    mP_Ar = np.zeros(velib.NOMBRE_ARR_PARIS)
    for sttn in sttns_info :
        arron = velib.get_arrondissement(sttn)
        mP_Ar[arron-1] = mP_Ar[arron-1] + 1

    return mP_Ar

def get_mP_Ar_hist(sttns_info):
    arrdmt = np.zeros(len(sttns_info))
    p = 0
    for sttn in sttns_info :
        arrdmt[p] = velib.get_arrondissement(sttn)
        p = p + 1

    nItervalles = 19
    rng = (1,20)
    res = plt.hist(arrdmt, nItervalles, rng)
    return res

def get_mP_Al(sttns_info):
    altitudes = np.zeros(len(sttns_info))
    nItervalles = 29
    
    p = 0
    for sttn in sttns_info :
        altitudes[p] = velib.get_alt(sttn)
        p = p + 1
    
    res = plt.hist(altitudes, nItervalles)
    # print res[0] # effectif dans les intervalles
    # print res[1] # definition des intervalles (ATTENTION: 31 valeurs)
    return res
        
def get_mP_Sp_Al(sttns_info):
    altitudes = np.zeros(len(sttns_info))
    nItervalles = 30
    
    p = 0
    for sttn in sttns_info :
        if velib.station_is_plein(sttn):
            altitudes[p] = velib.get_alt(sttn)
            p = p + 1
    p = p - 1
    res = plt.hist(altitudes[:p], nItervalles)
    return res
    
def get_mP_Vd_Al(sttns_info):
    altitudes = np.zeros(len(sttns_info))
    nItervalles = 30
    
    p = 0
    for sttn in sttns_info :
        if velib.station_velos_disp(sttn):
            altitudes[p] = velib.get_alt(sttn)
            p = p + 1
    p = p - 1
    res = plt.hist(altitudes[:p], nItervalles)
    return res
    
def get_mP_Vd_Ar(sttns_info):
    arrdmt = np.zeros(len(sttns_info))
    nItervalles = 20
    rng = (1,20)
    p = 0
    for sttn in sttns_info :
        if velib.station_velos_disp(sttn):
            arrdmt[p] = velib.get_arrondissement(sttn)            
            p = p + 1
    p = p - 1
    res = plt.hist(arrdmt[:p], nItervalles, rng)
    return res

def testcase_1(data):
    print "get_mP_Ar(data)"
    print get_mP_Ar(data)
    print "get_mP_Ar_hist(data)"
    print get_mP_Ar_hist(data)
    print "get_mP_Al(data)"
    print get_mP_Al(data)

def testcase_2(data):
    print "get_mP_Sp_Al(data)"
    print get_mP_Sp_Al(data)
    print "get_mP_Vd_Al(data)"
    print get_mP_Vd_Al(data)
    print "get_mP_Vd_Ar(data)"
    print get_mP_Vd_Ar(data)
    
def main():
    data = velib.load_velibs_info('dataVelib.pkl')
    velib.defiler_stations(data)
    velib.defiler_stations(data)
    velib.defiler_stations(data)

    print "case 1: "
    testcase_1(data)
    print "case 2: "
    testcase_2(data)
    
if __name__ == "__main__":
    main()
