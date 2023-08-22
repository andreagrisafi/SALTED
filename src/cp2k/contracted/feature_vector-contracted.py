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
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-j1", "--istart", type=int, default=0, help="starting index")
    parser.add_argument("-j2", "--iend",   type=int, default=0, help="ending index")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("dataset subselection")
# dataset slice boundaries 
istart = args.istart-1
iend = args.iend

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

kdir = inp.kerndir
fdir = inp.featdir

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

#kdir = {}
#rcuts = [6.0]
## get truncated size
#for rc in rcuts:
#    kdir[rc] = "kernels_rc"+str(rc)+"-sg"+str(rc/10)+"/"

#orcuts = np.loadtxt("optimal_rcuts.dat")

# compute the weight-vector size 
Mcut = {}
totsize = 0
iii=0
for spe in spelist:
    for l in range(lmax[spe]+1):
        for n in range(ncut[(spe,l)]):
            #orcuts[iii]
            #Mcut[(spe,l,n)] = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
            Mcut[(spe,l,n)] = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
            totsize += Mcut[(spe,l,n)]
            iii+=1

print("problem dimensionality:", totsize)


dirpath = os.path.join(inp.path2ml,fdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+fdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

#for iconf in range(ndata):
for iconf in range(istart,iend):

    start = time.time()
    print(iconf+1)

    # load reference QM data
    Tsize = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                Tsize += 2*l+1

    # initialize RKHS feature vectors for each channel 
    Psi = {}
    for spe in spelist:
        for l in range(lmax[spe]+1):
            lsize = natoms_per_spe[(iconf,spe)]*(2*l+1) 
            for n in range(ncut[(spe,l)]):
                Psi[(spe,l,n)] = np.zeros((lsize,totsize)) 

    # fill basis set dictionary of feature vectors to be diagonal in for each channel (spe,l,n)  
    ispe = {}
    isize = 0
    iii = 0
    for spe in spelist:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                #orcuts[iii]
                #psi_nm = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
                isize += Mcut
                iii += 1

    # fill in a single array for RKHS feature vector and predictions
    psi_vector = np.zeros((Tsize,totsize))
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            i1 = ispe[spe]*(2*l+1)
            i2 = ispe[spe]*(2*l+1)+2*l+1
            for n in range(ncut[(spe,l)]):
                psi_vector[i:i+2*l+1] = Psi[(spe,l,n)][i1:i2] 
                i += 2*l+1
        ispe[spe] += 1

    # save sparse feature-vector 
    nrows = psi_vector.shape[0]
    ncols = psi_vector.shape[1]
    srows = np.nonzero(psi_vector)[0]
    scols = np.nonzero(psi_vector)[1]
    ssize = len(srows)
    psi_nonzero = psi_vector[srows,scols] 
    ij = np.vstack((srows,scols))
    sparse_psi = sparse.coo_matrix((psi_nonzero, ij), shape=(nrows, ncols))
    sparse.save_npz(inp.path2ml+fdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz", sparse_psi)
 
    print(time.time()-start)
