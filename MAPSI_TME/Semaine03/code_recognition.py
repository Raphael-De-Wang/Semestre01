import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_file ( filename ):
    infile = open ( filename, "r" )
    nb_classes, nb_features = [  int(x) for x in infile.readline().strip().split() ]
    data = np.empty ( 10, dtype=object )
    filler = np.frompyfunc( lambda x: list(), 1, 1)
    filler( data, data )

    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( map ( lambda x: float(x), champs ) )
            
    infile.close ()

    output = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc( lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )
    
    return output

def display_image ( X ):    
    if X.size != 256:
        raise ValueError ( "Les images doivent etre de 16x16 pixels" )
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    img.shape = (16,16,3)
    plt.imshow( img )
    plt.show ()
    
def learnML_class_parameters ( classe ):
    mtx = np.zeros((len(classe.T),2))
    for i, pixte in enumerate(classe.T):
        mtx[i] = np.array([pixte.mean(), pixte.var()]) # deriver
    return mtx.T

def learnML_all_parameters ( data ):
    return [ learnML_class_parameters ( classe ) for classe in data ]

def log_loi_normale(x, mu, var):
    z = (( x - mu ) ** 2)/( 2 * var )
    y = math.sqrt( 2 * math.pi * var )
    return math.log(y, math.e) * (-1) - z
    
def log_likelihoods ( image, params ):
    lhs = np.ones(len(params))
    for i, param in enumerate( params ):
        for j, piste_param in enumerate( np.transpose(param) ):
            if piste_param[0] <> 0:
                lhs[i] += log_loi_normale(image[j], piste_param[0], piste_param[1])
    return lhs

def classify_image ( image, parameters ):
    return np.argmax( log_likelihoods ( image, parameters ) )

def classify_all_images ( data, parameters ):
    res = np.zeros((10,10))
    for i, classe in enumerate(data):
        for image in classe:
            res[i][ classify_image ( image, parameters ) ] += 1
    return res

def dessine ( classified_matrix ):
    print classified_matrix 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )
    plt.show()
    
def main():
    fname = '2014_tme3_usps_train.txt'
    data = read_file(fname)
    tfname = '2014_tme3_usps_test.txt'
    tdata = read_file(tfname)
    
    parameters = learnML_all_parameters ( data )
    '''
    print log_likelihoods ( image, parameters )
    print classify_image ( image, parameters )
    print classify_all_images ( tdata, parameters )
    '''
    dessine ( classify_all_images ( tdata, parameters ) )

if __name__ == "__main__":
    main()

