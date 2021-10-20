import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle
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
    for l in xrange(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut

# paths to data
kdir = inp.kerndir
pdir = inp.preddir

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

# compute the weight-vector size 
Mcut = {}
totsize = 0
for spe in spelist:
    for l in xrange(lmax[spe]+1):
        for n in xrange(nmax[(spe,l)]):
            Mcut[(spe,l,n)] = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
            totsize += Mcut[(spe,l,n)]

print "problem dimensionality:", totsize


dirpath = os.path.join(inp.path2ml, "psi-vectors")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+"psi-vectors/", "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

#for iconf in xrange(ndata):
for iconf in xrange(istart,iend):

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
    for spe in spelist:
        ispe[spe] = 0
        for l in xrange(lmax[spe]+1):
            psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
            Mcut = psi_nm.shape[1]
            for n in xrange(nmax[(spe,l)]):
                Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
                isize += Mcut

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

    np.save(inp.path2ml+"psi-vectors/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_vector)

    print time.time()-start



