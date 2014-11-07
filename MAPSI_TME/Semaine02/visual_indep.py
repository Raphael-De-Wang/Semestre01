import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as stats

from mpl_toolkits.mplot3d import Axes3D

def loi_normal_centree(k, sigma):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    arr = np.arange(0,k,1)
    v_arr = stats.norm(k>>1,sigma).pdf(arr)
    return arr, v_arr

def proba_affine (k, slope):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs ( slope  ) > 2.0 / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + str(2/(k*k)))
        
    arr = np.arange(0, k, 1)
    v_arr = np.zeros(k)
    for i in arr:
        v_arr[i] = 1/k + slope * i + slope * ( k - 1 ) / 2
        
    return arr, v_arr

def dessine_2D(X, Y):
    pl.plot(X, Y, label="$X$", color="blue")
    pl.show()

def Pxy (PA, PB):
    m_PA = np.tile(PA, (len(PA),1)).T
    m_PB = np.tile(PB, (len(PB),1))
    print m_PA
    print m_PB
    return m_PB * m_PA

def dessine_3D ( P_jointe ):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    pl.show()
    
def testcase_Pxy():
    PA = np.array ( [0.2, 0.7, 0.1] )
    PB = np.array ( [0.4, 0.4, 0.2] )
    print Pxy (PA, PB)

def testcase_dessine_2D():
    k = 21
    sigma = 5
    slope = 0.0032
    # X, Y = loi_normal_centree(k, sigma)
    X, Y = proba_affine(k, slope)
    dessine_2D(X, Y)
    
def testcase_dessine_3D():
    X_norm, Y_norm = loi_normal_centree(21, 5)
    X_affi, Y_affi = proba_affine (21, 0.0032)
    P_jointe = Pxy(Y_affi,Y_norm)
    print P_jointe
    dessine_3D(P_jointe)
    
def main():
    # testcase_dessine_2D()
    # testcase_Pxy()
    testcase_dessine_3D()
    
if __name__ == "__main__":
    main()
