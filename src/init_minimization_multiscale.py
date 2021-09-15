import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle

from scipy.optimize import minimize

import basis

sys.path.insert(0, './')
import inp

# system definition
spelist = inp.species
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# basis definition
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in xrange(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut

# species dependent arrays
atoms_per_spe = {}
natoms_per_spe = {}
for iconf in xrange(ndata):
    for spe in spelist:
        atoms_per_spe[(iconf,spe)] = []
        natoms_per_spe[(iconf,spe)] = 0

atomic_symbols = []
valences = []
natoms = np.zeros(ndata,int)
for iconf in xrange(ndata):
    atomic_symbols.append(xyzfile[iconf].get_chemical_symbols())
    valences.append(xyzfile[iconf].get_atomic_numbers())
    natoms[iconf] = int(len(atomic_symbols[iconf]))
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        atoms_per_spe[(iconf,spe)].append(iat)
        natoms_per_spe[(iconf,spe)] += 1
natmax = max(natoms)

# load average density coefficients
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

kdir = {}
rcuts = [2.0,3.0,4.0,5.0,6.0]
# get truncated size
for rc in rcuts:
    kdir[rc] = "kernels_rc"+str(rc)+"-sg"+str(rc/10)+"/"

orcuts = np.loadtxt("optimal_rcuts.dat")

# compute the weight-vector size 
Mcut = {}
totsize = 0
iii=0
for spe in spelist:
    for l in xrange(lmax[spe]+1):
        for n in xrange(nmax[(spe,l)]):
            rc = orcuts[iii]
            Mcut[(spe,l,n)] = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
            totsize += Mcut[(spe,l,n)]
            iii+=1

print "problem dimensionality:", totsize


dirpath = os.path.join(inp.path2ml, "psi-vectors_multiscale")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+"psi-vectors_multiscale/", "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

for iconf in xrange(ndata):


    start = time.time()
    print iconf
    # load reference QM data
    overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    Tsize = len(overl)
   
    # initialize RKHS feature vectors for each channel 
    Psi = {}
    for spe in spelist:
        for l in xrange(lmax[spe]+1):
            lsize = natoms_per_spe[(iconf,spe)]*(2*l+1) 
            for n in xrange(nmax[(spe,l)]):
                Psi[(spe,l,n)] = np.zeros((lsize,totsize)) 

    # load the RKHS feature vectors and compute predictions for each channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    for spe in spelist:
        ispe[spe] = 0
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                rc = orcuts[iii]
                psi_nm = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
                isize += Mcut
                iii += 1

    # fill in a single array for RKHS feature vector and predictions
    psi_vector = np.zeros((Tsize,totsize))
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in xrange(lmax[spe]+1):
            i1 = ispe[spe]*(2*l+1)
            i2 = ispe[spe]*(2*l+1)+2*l+1
            for n in xrange(nmax[(spe,l)]):
                psi_vector[i:i+2*l+1] = Psi[(spe,l,n)][i1:i2] 
                i += 2*l+1
        ispe[spe] += 1

    np.save(inp.path2ml+"psi-vectors_multiscale/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_vector)

    print time.time()-start



