#!/usr/bin/python

import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-s",   "--testset",     type=int,   default=1,   help="test dataset selection")
    parser.add_argument("-f",   "--trainfrac"  , type=float, default=1.0, help="training set fraction")
    parser.add_argument("-m",   "--msize"  ,     type=int,   default=100, help="number of reference environments")
    parser.add_argument("-rc",  "--cutoffradius"  , type=float, default=4.0, help="soap cutoff")
    parser.add_argument("-sg",  "--sigmasoap"  , type=float, default=0.3, help="soap sigma")
    parser.add_argument("-r",   "--regular"  ,   type=float, default=1e-06, help="regularization")
    parser.add_argument("-jit", "--jitter"  ,   type=float, default=1e-10, help="jitter")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    s = args.testset
    f = args.trainfrac
    m = args.msize
    rc = args.cutoffradius
    sg = args.sigmasoap
    r = args.regular  
    jit = args.jitter
    return [s,f,m,rc,sg,r,jit]

args = add_command_line_arguments_contraction("density regression")
[nset,frac,M,rc,sigma_soap,reg,jit] = set_variable_values_contraction(args)

# coversion factors
bohr2ang = 0.529177249

# system definition
filename = "coords_1000.xyz"
xyzfile = read(filename,":")
ndata = len(xyzfile)

# system parameters
coords = []
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    coords.append(np.asarray(xyzfile[i].get_positions(),float)/bohr2ang)
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)

#================== species dictionary
spe_dict = {}
spe_dict[0] = "H"
spe_dict[1] = "O"
#====================================== reference environments 
fps_indexes = np.loadtxt("SELECTIONS/refs_selection_"+str(M)+".txt",int)
fps_species = np.loadtxt("SELECTIONS/spec_selection_"+str(M)+".txt",int)
#============== angular 
lmax = {}
llmax = 3
lmax["O"] = 3
lmax["H"] = 2
nnmax = 10
nmax = {}
# oxygen
nmax[("O",0)] = 10
nmax[("O",1)] = 7
nmax[("O",2)] = 5
nmax[("O",3)] = 2
# hydrogen
nmax[("H",0)] = 4
nmax[("H",1)] = 3
nmax[("H",2)] = 2


# basis set arrays 
bsize = np.zeros(nspecies,int)
almax = np.zeros(nspecies,int)
anmax = np.zeros((nspecies,llmax+1),int)
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    almax[ispe] = lmax[spe]+1
    for l in xrange(lmax[spe]+1):
        anmax[ispe,l] = nmax[(spe,l)]
        bsize[ispe] += nmax[(spe,l)]*(2*l+1)

# problem dimensionality 
collsize = np.zeros(M,int)
for iref in xrange(1,M):
    collsize[iref] = collsize[iref-1] + bsize[fps_species[iref-1]]
totsize = collsize[-1] + bsize[fps_species[-1]]
print "problem dimensionality =", totsize

Avec = np.load("MATRICES/Avec_M"+str(M)+"_trainfrac"+str(frac)+".npy")
Bmat = np.load("MATRICES/Bmat_M"+str(M)+"_trainfrac"+str(frac)+".npy")
Rmat = np.load("MATRICES/KMM_"+str(M)+".npy")

# solve the regularized sparse regression problem 
weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)

np.save("WEIGHTS/weights_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",weights)
