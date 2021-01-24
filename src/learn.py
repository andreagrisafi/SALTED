import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read

import basis
sys.path.insert(0, './')
import inp

# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i]

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

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# number of sparse environments
M = inp.Menv

# number of sparse environments
reg = inp.regul

# number of sparse environments
jit = inp.jitter

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
start = time.time()
weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)
#weights = np.linalg.lstsq(Bmat,Avec,rcond=None)[0]
print time.time() - start, "seconds"

# save
np.save("weights.npy",weights)
