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
    parser.add_argument("-p", "--partial", type=int, default=0, help="Calculate A and B for the pth ten structures in the training set")

    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("density regression")
reg = args.regul
p = args.partial
xv = inp.xv

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
if reg is None: reg = inp.regul

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

iters = 1
if xv: iters = 2

for count in range(iters):

    print "Loading regression matrices ..."
    if p > 0:
       for i in range(1,p+1):
           print i

           if count == 0:
               Avec_p = np.load(inp.path2data+"matrices/A_"+str(i)+"_vector.npy")
               Bmat_p = np.load(inp.path2data+"matrices/B_"+str(i)+"_matrix.npy")
           elif count == 1:
               Avec_p = np.load(inp.path2data+"matrices/Ap_"+str(i)+"_vector.npy")
               Bmat_p = np.load(inp.path2data+"matrices/Bp_"+str(i)+"_matrix.npy")
           if i == 1:
               Avec = Avec_p.copy()
               Bmat = Bmat_p.copy()
           else:
               Avec += Avec_p
               Bmat += Bmat_p

    else:
        if count == 0:
           Avec = np.load(inp.path2data+"matrices/A_vector.npy")
           Bmat = np.load(inp.path2data+"matrices/B_matrix.npy")
        else:
           Avec = np.load(inp.path2data+"matrices/Ap_vector.npy")
           Bmat = np.load(inp.path2data+"matrices/Bp_matrix.npy")
    
    Rmat = np.load("Kmm_matrix.npy")

    print "Solving regression problem of dimension =", totsize
    start = time.time()
    #weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)
    weights = np.linalg.lstsq(Bmat+reg*Rmat,Avec,rcond=None)[0]
    print time.time() - start, "seconds"

    # save
    if count == 0:
        np.save("weights.npy",weights)
    elif count == 1:
        np.save("weights_p.npy",weights)
