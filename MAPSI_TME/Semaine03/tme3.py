import numpy as np
import matplotlib.pyplot as plt
import math as math
from mpl_toolkits.mplot3d import Axes3D

def read_file ( filename ):
	"""
	Lit un fichier USPS et renvoie un tableau de tableaux d'images.
	Chaque image est un tableau de nombres reels.
	Chaque tableau d'images contient des images de la meme classe.
	Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
	des images de la classe 0, T[1] contient celui des images de la classe 1,
	et ainsi de suite.
	"""
	# lecture de l'en-tete
	infile = open ( filename, "r" )    
	nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

	# creation de la structure de donnees pour sauver les images :
	# c'est un tableau de listes (1 par classe)
	data = np.empty ( 10, dtype=object )  
	filler = np.frompyfunc(lambda x: list(), 1, 1)
	filler( data, data )

	# lecture des images du fichier et tri, classe par classe
	for ligne in infile:
		champs = ligne.split ()
		if len ( champs ) == nb_features + 1:
			classe = int ( champs.pop ( 0 ) )
			data[classe].append ( map ( lambda x: float(x), champs ) )    
	infile.close ()

	# transformation des list en array
	output  = np.empty ( 10, dtype=object )
	filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
	filler2 ( data, output )

	return output
	
# lecture des fichiers
training_data= read_file("usps_train")
test_data=read_file("usps_test")

def display_image ( X ):
	"""
	Etant donne un tableau de 256 flotants representant une image de 16x16
	pixels, la fonction affiche cette image dans une fenetre.
	"""
	# on teste que le tableau contient bien 256 valeurs
	if X.size != 256:
		raise ValueError ( "Les images doivent etre de 16x16 pixels" )

	# on cree une image pour imshow: chaque pixel est un tableau a 3 valeurs
	# (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
	Y = X / X.max ()
	img = np.zeros ( ( Y.size, 3 ) )
	for i in range ( 3 ):
		img[:,i] = X

	# on indique que toutes les images sont de 16x16 pixels
	img.shape = (16,16,3)

	# affichage de l'image
	plt.imshow( img )
	plt.show ()

#display_image(training_data[4][2])



def learnML_class_parameters ( classe ):

	xij=0
	n=len(classe)
	mu=np.zeros(256)
	sigma=np.zeros(256)
	xi=np.zeros(n)
	# le pixel i et l'image j
	for i in range(256): # parcourir des pixels dans l'image
		j=0
		for image in classe: # parcourir des images de cette classe
			xi[j]= image[i]
			j+=1
		mu[i]=xi.sum()/n
		for j in xi:
			sigma[i]+=(j-mu[i])*(j-mu[i])
		sigma[i] /= n-1
            #		print mu[i],sigma[i]
	return (mu,sigma)
(mu,sigma)=learnML_class_parameters (training_data[5])
#print len(mu),len(sigma)


def learnML_all_parameters ( train_data ):
	parametres=[]
	for classe in train_data:
		parametres.append(learnML_class_parameters (classe))
	return parametres

parameters=learnML_all_parameters ( training_data )
#print parametres


def log_likelihoods ( image, parameters ):
	
	n=len(parameters)
	tab=np.zeros(n)
	k=0
	for (mu,sigma) in parameters:
		for i in range(256):
			if sigma[i]!=0.0:
				lv=(image[i]-mu[i])*(mu[i]-image[i])/sigma[i]/2. - math.log(2.*math.pi*sigma[i])/2.
				tab[k]+=lv
		k+=1
	return tab

print log_likelihoods(test_data[2][2],parameters)


def classify_image ( image, parameters ):
	tab=log_likelihoods(image,parameters)
	mx=tab[0]
	ind=0
	for i in range(1,len(tab)):
		if tab[i]>mx:
			mx=tab[i]
			ind=i
	return ind

def classify_all_images ( test_data, parameters ):
	n=len(parameters)
	T=np.eye(n)*0.
	i=0
	for classe in test_data:
		for image in classe:
			j=classify_image(image,parameters)
			T[i][j]+=1.
		T[i]/=len(classe)
		i+=1
	return T



#T=classify_all_images ( test_data, parameters )


def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )
    plt.show()

#dessine(T)


