import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse

sys.path.insert(0, './')
import inp

sys.path.insert(0, '../../src/')
import basis

# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i]

# read basis
[llmax,lmax,nnmax,nmax] = basis.basiset(inp.basis)

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# number of sparse environments
M = inp.Menv

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r",   "--regular"  ,   type=float, default=1e-06, help="regularization")
    parser.add_argument("-jit", "--jitter"  ,   type=float, default=1e-10, help="jitter")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    r = args.regular  
    jit = args.jitter
    return [r,jit]

args = add_command_line_arguments_contraction("density regression")
[reg,jit] = set_variable_values_contraction(args)

# system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)

#====================================== reference environments 
fps_indexes = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,1]

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

print "Loading regression matrices ..."
Avec = np.load("A_vector.npy")
Bmat = np.load("B_matrix.npy")
Rmat = np.load("Kmm_matrix.npy")

print "Solving regression problem of dimension =", totsize
weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)

# save
np.save("weights.npy",weights)
