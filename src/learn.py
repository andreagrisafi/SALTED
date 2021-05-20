import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse

import basis
sys.path.insert(0, './')
import inp

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r", "--regul", type=float, default=None, help="Read regularization parameter at run-time")
    parser.add_argument("-np", "--number_partial", type=int, default=0, help="Build A and B from the first p blocks of structures in the training set")

    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("density regression")
# Read the regularization parameter at runtime (this will override inp.py)
reg = args.regul

# Read the number of blocks of structures to include in the regression matrix and vector
p = args.number_partial

# Is automatic cross-validation requested?
xv = inp.xv

# Use Singular Value Decomposition to find the regression weights
svd = inp.svd
if xv and (p > 0): sys.exit('The options xv and -np are not compatible')

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

if reg is None: reg = inp.regul

if not svd: jit = inp.jitter

# system parameters
atomic_symbols = []
atomic_valence = []
print ndata
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

for i in range(2):
    # Two loops only performed when cross-validating
    if i == 1 and not xv: continue

    print "Loading regression matrices ..."

    # Include the first np blocks of structures in A and B
    if p > 0:
       for j in range(1,p+1):

           Avec_p = np.load(inp.path2data+"matrices/A_"+str(j)+"_vector.npy")
           Bmat_p = np.load(inp.path2data+"matrices/B_"+str(j)+"_matrix.npy")

           if j == 1:
               Avec = Avec_p.copy()
               Bmat = Bmat_p.copy()
           else:
               Avec += Avec_p
               Bmat += Bmat_p

    else:
        if i == 0:
           Avec = np.load(inp.path2data+"matrices/A_vector.npy")
           Bmat = np.load(inp.path2data+"matrices/B_matrix.npy")
        else:
           Avec = np.load(inp.path2data+"matrices/Ap_vector.npy")
           Bmat = np.load(inp.path2data+"matrices/Bp_matrix.npy")
    
    Rmat = np.load("Kmm_matrix.npy")

    print "Solving regression problem of dimension =", totsize
    start = time.time()
    if svd:
        weights = np.linalg.lstsq(Bmat+reg*Rmat,Avec,rcond=None)[0]
    else:
        weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)
    print time.time() - start, "seconds"

    # save
    if i == 0:
        np.save("weights.npy",weights)
    elif i == 1:
        np.save("weights_p.npy",weights)
