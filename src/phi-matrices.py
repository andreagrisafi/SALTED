import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle
import random

import basis

sys.path.insert(0, './')
import inp

spelist = inp.species
# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

# number of sparse environments
M = inp.Menv

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))

# initialize useful dictionaries
A = {}
B = {}
Mcut = {}
av_coefs = {}
nenv = {}
totsize = 0
for spe in spelist:
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)
    nenv[spe] = 0
    for l in xrange(lmax[spe]+1):
        Mcut[(spe,l)] = np.load(inp.path2ml+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(0)+"_M"+str(M)+".npy").shape[1]
        for n in xrange(nmax[(spe,l)]):
            totsize += Mcut[(spe,l)]
            A[(spe,l,n)] = np.zeros(Mcut[(spe,l)])
            for spe2 in spelist:
                for l2 in xrange(lmax[spe2]+1):
                    Mcut[(spe2,l2)] = np.load(inp.path2ml+"kernels/spe"+str(spe2)+"_l"+str(l2)+"/psi-nm_conf"+str(0)+"_M"+str(M)+".npy").shape[1]
                    for n2 in xrange(nmax[(spe2,l2)]):
                        B[(spe,l,n,spe2,l2,n2)] = np.zeros((Mcut[(spe,l)],Mcut[(spe2,l2)]))

print "problem dimensionality =", totsize
print ""

for iconf in xrange(ndata):
    Coef = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
    #Proj = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
    #Over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    #Coef = np.linalg.solve(Over,Proj)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        nenv[spe] += 1
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       av_coefs[spe][n] += Coef[i]
                    i += 1

print "computing mean spherical averages..."
for spe in spelist:
    av_coefs[spe] /= nenv[spe]
    np.save("averages_"+str(spe)+".npy",av_coefs[spe])

dataset = range(ndata)
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
ntrain = int(inp.trainfrac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]

print "collecting contributions from training structures..."
print ""
for iconf in trainrange:
    print iconf

    # compute S^{1/2} cutting small/negative eigenvalues
    S = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    eva, eve = np.linalg.eigh(S)
    eva = eva[eva>1e-08]
    eve = eve[:,-len(eva):]
    Ssqrt = np.dot(eve,np.diag(np.sqrt(eva)))
  
    # fill array of spherical averages
    Av_coeffs = np.zeros(len(S))
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 1
    
    # center density projections about spherical averages
    P = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
    P -= np.dot(S,Av_coeffs)

    # load Phi-vectors
    psi_nm = {}
    ispe = {}
    for spe in spelist:
        ispe[spe] = 0
        for l in xrange(lmax[spe]+1):
            psi_nm[(spe,l)] = np.load(inp.path2ml+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(iconf)+"_M"+str(M)+".npy")

    # compute A-vector and B-matrix
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat] 
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                A[(spe,l,n)] += np.dot(psi_nm[(spe,l)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1].T,P[i:i+2*l+1])
                Psi1 = np.dot(psi_nm[(spe,l)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1].T,Ssqrt[i:i+2*l+1])
                
                ispe2 = {}
                for spe2 in spelist:
                    ispe2[spe2] = 0

                i2 = 0
                for jat in xrange(natoms[iconf]):
                    spe2 = atomic_symbols[iconf][jat] 
                    for l2 in xrange(lmax[spe2]+1):
                        for n2 in xrange(nmax[(spe2,l2)]):
                            Psi2 = np.dot(psi_nm[(spe2,l2)][ispe2[spe2]*(2*l2+1):ispe2[spe2]*(2*l2+1)+2*l2+1].T,Ssqrt[i2:i2+2*l2+1])
                            B[(spe,l,n,spe2,l2,n2)] += np.dot(Psi1,Psi2.T)
                            i2 += 2*l2+1
                    ispe2[spe2] += 1

                i += 2*l+1
        ispe[spe] += 1

# fill arrays for A-vector and B-matrix
Avec = np.zeros(totsize)
Bmat = np.zeros((totsize,totsize))
isize = 0
for spe in spelist:
    for l in xrange(lmax[spe]+1):
        for n in xrange(nmax[(spe,l)]):
            Avec[isize:isize+Mcut[(spe,l)]] = A[(spe,l,n)] 
            isize2 = 0
            for spe2 in spelist:
                for l2 in xrange(lmax[spe2]+1):
                    for n2 in xrange(nmax[(spe2,l2)]):
                        Bmat[isize:isize+Mcut[(spe,l)],isize2:isize2+Mcut[(spe2,l2)]] = B[(spe,l,n,spe2,l2,n2)] 
                        isize2 += Mcut[(spe2,l2)]
            isize += Mcut[(spe,l)]

np.save("Avec.npy",Avec)
np.save("Bmat.npy",Bmat)
