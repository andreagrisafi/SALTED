import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle
from scipy import sparse

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
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

pdir = inp.preddir
coefdir = inp.coefdir
regrdir = inp.regrdir
featdir = inp.featdir

# species dependent arrays
atoms_per_spe = {}
natoms_per_spe = {}
for iconf in range(ndata):
    for spe in spelist:
        atoms_per_spe[(iconf,spe)] = []
        natoms_per_spe[(iconf,spe)] = 0

atomic_symbols = []
valences = []
natoms = np.zeros(ndata,int)
for iconf in range(ndata):
    atomic_symbols.append(xyzfile[iconf].get_chemical_symbols())
    valences.append(xyzfile[iconf].get_atomic_numbers())
    natoms[iconf] = int(len(atomic_symbols[iconf]))
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        atoms_per_spe[(iconf,spe)].append(iat)
        natoms_per_spe[(iconf,spe)] += 1
natmax = max(natoms)

projector = {}
ncut = {}
for spe in spelist:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]

p = sparse.load_npz(inp.path2ml+featdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf0.npz")
totsize = p.shape[-1]
print("problem dimensionality:", totsize,flush=True)
if totsize>70000:
    print("ERROR: problem dimension too large, minimize directly loss-function instead!")
    sys.exit(0)

# load average density coefficients
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# define training set at random
dataset = list(range(ndata))
#random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set_N"+str(inp.Ntrain)+".txt",trainrangetot,fmt='%i')
#trainrangetot = np.loadtxt("training_set_upto443.txt",int)
#trainrangetot = np.loadtxt("training_set_upto663.txt",int)

# Distribute structures to tasks
ntrain = int(inp.trainfrac*inp.Ntrain)
trainrange = trainrangetot[:ntrain]

dirpath = os.path.join(inp.path2ml, regrdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+regrdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

print("computing regression matrices...")
Avec = np.zeros(totsize)
Bmat = np.zeros((totsize,totsize))
#Avec = np.load(inp.path2ml+regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N90.npy")*90
#Bmat = np.load(inp.path2ml+regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N90.npy")*90
for iconf in trainrange:
    print("conf:", iconf+1,flush=True)
    
    # load reference QM data
    ref_coefs = np.load(inp.path2qm+coefdir+"coefficients_conf"+str(iconf)+".npy")
    over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    psivec = sparse.load_npz(inp.path2ml+featdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz")
    psi = psivec.toarray()

    #TODO comment this to test learning of difference!
    Av_coeffs = np.zeros(ref_coefs.shape[0])
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                if l==0:
                   Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
    
    ref_coefs -= Av_coeffs
    ref_projs = np.dot(over,ref_coefs)
    
    Avec += np.dot(psi.T,ref_projs)
    Bmat += np.dot(psi.T,np.dot(over,psi))

    if iconf+1==4 or iconf+1==8 or iconf+1==16 or iconf+1==32 or iconf+1==39: 
        np.save(inp.path2ml+regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(iconf+1)+".npy",Avec/float(iconf+1))
        np.save(inp.path2ml+regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(iconf+1)+".npy",Bmat/float(iconf+1))

Avec /= float(ntrain)
Bmat /= float(ntrain)

np.save(inp.path2ml+regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(ntrain)+".npy",Avec)
np.save(inp.path2ml+regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(ntrain)+".npy",Bmat)
