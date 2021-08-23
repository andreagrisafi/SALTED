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

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in xrange(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
kdir = inp.kerndir
pdir = inp.preddir

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

#Avec = np.load("Avec.npy")
#Bmat = np.load("Bmat.npy")
#Msize = len(Avec)
#weights = np.linalg.solve(Bmat+inp.regul*np.eye(Msize),Avec)

weights = np.load("weights.npy")
Msize = len(weights)

trainrangetot = np.loadtxt("training_set.txt",int)
testrange = np.setdiff1d(range(ndata),trainrangetot)

itest = 0
error_density = 0
variance = 0
coeffs = np.zeros((len(testrange),natmax,llmax+1,nnmax,2*llmax+1))
for iconf in testrange:

    # load reference
    ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
    ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
    overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    Tsize = len(ref_coefs)

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    for spe in spelist:
        ispe[spe] = 0
        for l in xrange(lmax[spe]+1):
            psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
            Mcut = psi_nm.shape[1]
            for n in xrange(nmax[(spe,l)]):
                C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                isize += Mcut
        
    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                if l==0:
                   Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1

    # rebuild predictions
    pred_coefs += Av_coeffs
    pred_projs = np.dot(overl,pred_coefs)
    np.save(inp.path2qm+pdir+"prediction_conf"+str(iconf)+".npy",pred_projs)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    coeffs[itest,iat,l,n,im] = pred_coefs[i]
                    i += 1
    
    # compute error
    error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
    error_density += error
    ref_projs -= np.dot(overl,Av_coeffs)
    ref_coefs -= Av_coeffs
    var = np.dot(ref_coefs,ref_projs)
    variance += var
    print iconf+1, ":", np.sqrt(error/var)*100, "% RMSE"
    itest+=1

print ""
print "% RMSE =", 100*np.sqrt(error_density/variance)

np.save("pred_coeffs.npy",coeffs)
