import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl

import dist_de_prob as ddp
import velib


DISPONIBLE       = 1
NON_DISPONBILE   = 0
def matrix_vd_al(sttns_info):
    mtx = np.zeros((2, len(sttns_info)),int)
    p = 0
    for sttn in sttns_info:
        if velib.station_velos_disp(sttn):
            mtx[0][p] = 1
        mtx[1][p] = np.floor(velib.get_alt(sttn))
        p += 1

    return mtx

def trier_vd_al_by_vd(m_vd_al, vd_key):
    # key 0 ou 1
    return [m_vd_al[1][x] for x in range(len(m_vd_al[0])) if m_vd_al[0][x] == vd_key]

def trier_vd_al_by_al(m_vd_al, al_key):
    return [m_vd_al[0][x] for x in range(len(m_vd_al[0])) if m_vd_al[1][x] == al_key]
    
def classer_al(altitudes, bins):
    # intervalle = np.ceil((altitudes.max()-altitudes.min())/bins)
    return np.floor((altitudes - altitudes.min())*bins/(altitudes.max()-altitudes.min()))

def cal_freq_vd(mtx_vd_al, vd_key):
    m_freq = np.zeros(len(np.unique(mtx_vd_al[1])))
    for p in range(len(m_freq)):
        vd_l    = trier_vd_al_by_al(mtx_vd_al, p)
        vd_num  = sum(vd_l)
        pos_num = len(vd_l)
        if pos_num == 0:
            m_freq[p] = -1
            continue
        if vd_key == DISPONIBLE:
            m_freq[p] = float(vd_num) / pos_num
        else:
            m_freq[p] = float(pos_num - vd_num) / pos_num
            
    return m_freq

def trace_vd_freq():
    return 0
    
def testcase():
    data = velib.load_velibs_info('dataVelib.pkl')
    velib.defiler_stations(data)
    velib.defiler_stations(data)
    velib.defiler_stations(data)
    
    mtx =  matrix_vd_al(data)
    vd_al = trier_vd_al_by_vd(mtx, DISPONIBLE)
    # print len(plt.hist(vd_al, 30)[0])
    mtx[1] = classer_al(mtx[1], 30)
    print trier_vd_al_by_al(mtx, 10)
    print cal_freq_vd(mtx, 0)
    
def main():
    testcase()
    
if __name__ == "__main__":
    main()
