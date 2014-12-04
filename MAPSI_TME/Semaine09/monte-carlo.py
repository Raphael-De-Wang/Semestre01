# -*- coding: utf-8 -*-

import math
import numpy as np
import pickle as pkl
import numpy.random as npr
import matplotlib.pyplot as pl

def tirage (m,N=1) :
    if N <= 1:
        return npr.uniform(-m,m,2)
    else:
        return npr.uniform(-m,m,(N,2))

def f(x,y,m):
    if x**2 + y**2 <= m**2 :
        return 1
    return 0

def PI(tirs):
    return 4.0 * sum([ f(x,y,1) for [x,y] in tirs ]) / len(tirs)

def monteCarlo (N) :
    tirs = np.array(tirage (1,N))
    return PI(tirs, N), tirs[:,0], tirs[:,1]

def MCMC (m, burnin, mixin, N):
    tirs = tirage (1,N)
    mcmc_tirs = [ pos for i, pos in enumerate(tirs) if i > burnin and i%mixin == 0 ]
    return PI(mcmc_tirs)

def dessine():
    pl.figure()
    pl.plot([-1,-1,1,-1],[-1,1,1,-1],'-');
    x = np.linspace(-1,1,100);
    y = np.sqrt(1-x*x);
    pl.plot(x,y,'b');
    pl.plot(x,-y,'b');

    pi, x, y = monteCarlo(int(1e4));

    dist = x*x + y*y
    pl.plot(x[dist<=1], y[dist<=1], "go")
    pl.plot(x[dist>1], y[dist>1],   "ro")
    pl.show()

# Décodage par la méthode de Metropolis-Hastings
(count, mu, A) = pkl.load(file("countWar.pkl", "rb"))
secret  = (open("secret.txt","r").read())[0:-1]
# secret2 = (open("secret2.txt","r").read())[0:-1]
chars = dict(zip(range(len(count)), count.keys()))
'''
print count.keys()
print np.shape(mu)
print np.shape(A)
#print secret
'''

def swapF(f):
    hasard = [0,0]
    while hasard[0] == hasard[1]:
        hasard = np.floor(npr.rand(2)*len(f))
    keys = f.keys()
    keys.sort()
    k1 = keys[int(hasard[0])]
    k2 = keys[int(hasard[1])]
    f[k1],f[k2] = f[k2],f[k1]

'''
f = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5}
swapF(f)
print f
'''

def decrypt(m,f):
    return [ f[c] for c in m ]

def logLikelihood(m,mu,A,chars,f):
    x = np.array(list(set(m)))
    llh = math.log(mu[f[x[0]]])
    for i in range(len(m) - 1 ):
        llh += math.log(A[f[m[i]],f[m[i+1]]])
    return llh

def MetropolisHastings(m,mu,A,f,N,chars):
    for i in range(N):
        fp = dict(f)
        # tirage de f' grâce à swapF à partir de la fonction courante f
        swapF(fp)
        
        # calcul de la log-vraisemblance du message décodée grâce à f'
        vrais = logLikelihood(m,mu,A,chars,fp)

        # tirage pour accepter ou non la transition vers f' selon le rapport des vraisemblances
        if i == 0:
            max_vrais = vrais
        else:
            # si la transition est acceptée, sauvegarder le message décodé avec la plus haute vraisemblance.
            if vrais > max_vrais:
                max_vrais = vrais
                f = fp

    return f

def identityF(keys):
   f = {}
   for k in keys:
      f[k] = k
   return f

def replaceF(f, kM, k):
   try:
      for c in f.keys():
         if f[c] == k:
            f[c] = f[kM]
            f[kM] = k
            return
   except KeyError as e:
      f[kM] = k

def mostFrequentF(message, count1, f={}):
   count = dict(count1)
   countM = {}
   # updateOccurrences(message, countM)
   while len(countM) > 0:
      bestKM = mostFrequent(countM)
      bestK = mostFrequent(count)
      if len(bestKM)==1:
         kM = bestKM[0]
      else:
         kM = bestKM[npr.random_integers(0, len(bestKM)-1)]
      if len(bestK)==1:
         k = bestK[0]
      else:
         k = bestK[npr.random_integers(0, len(bestK)-1)]
      replaceF(f, kM, k) 
      countM.pop(kM)
      count.pop(k)
   return f

fInit = identityF(count.keys())
fInit = mostFrequentF(secret, count, fInit)

MetropolisHastings(secret, mu, A, fInit, int(5e4), chars)
