import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle
from scipy import special
import random

import basis

sys.path.insert(0, './')
import inp

spelist = inp.species
# read system
xyzfile = read(inp.filenametest,":")
ndata = len(xyzfile)

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

kdir = inp.kerndirtest
kdir = inp.kerndirtest
rdir = inp.regrdir

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in range(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

dirpath = os.path.join(inp.path2ml, pdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+pdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

ntrain = inp.Ntrain

dirpath = os.path.join(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N_"+str(ntrain))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# load regression weights
weights = np.load(inp.path2ml+rdir+"weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")

# compute error over test set
error_density = 0
variance = 0
for itest in range(ndata):

    Tsize = 0
    for iat in range(natoms[itest]):
        spe = atomic_symbols[itest][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Tsize += 2*l+1

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    for spe in spelist:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                #rc = 6.0#orcuts[iii]
                #psi_nm = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(itest)+".npy") 
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(itest)+".npy") 
                Mcut = psi_nm.shape[1]
                C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                isize += Mcut
                iii += 1
    
        
    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in range(natoms[itest]):
        spe = atomic_symbols[itest][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                if l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1

    # add the average spherical coefficients to the predictions 
    pred_coefs += Av_coeffs

    # save predicted coefficients
    np.save(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(itest)+".npy",pred_coefs)
