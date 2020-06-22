import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse
import random

import basis

from lib import matrices 

sys.path.insert(0, './')
import inp


# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i]

# read basis
[lmax,nmax] = basis.basiset(inp.basis)

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

# number of training configurations 
N = inp.Ntrain

# training set fraction
frac = inp.trainfrac

# system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

# atomic species arrays
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)
spec_list = []
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_valence[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list.append(ispe)
               spec_list_per_conf[iconf].append(ispe)
spec_array = np.asarray(spec_list,int)
nenv = len(spec_array)

# atomic indexes sorted by valence 
atomicindx = np.zeros((natmax,nspecies,ndata),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[icount,ispe,iconf] = indexes[icount]


#====================================== reference environments 
fps_indexes = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,1]

# basis set size 
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

# training set selection
dataset = range(ndata)
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:N]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
print "Number of training configurations =", ntrain

# training set arrays
train_configs = np.array(trainrange,int)
atomicindx_training = atomicindx[:,:,trainrange]
atom_counting_training = atom_counting[trainrange] 
atomic_species = np.zeros((ntrain,natmax),int)
for itrain in xrange(ntrain):
    for iat in xrange(natoms_train[itrain]):
        atomic_species[itrain,iat] = spec_list_per_conf[trainrange[itrain]][iat]

# sparse overlap and projection indexes 
total_sizes = np.zeros(ntrain,int)
itrain = 0
for iconf in trainrange:
    atoms = atomic_symbols[iconf]
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            msize = 2*l+1
            for n in xrange(nmax[(atoms[iat],l)]):
                for im in xrange(msize):
                    total_sizes[itrain] += 1
    itrain += 1

# sparse kernel indexes 
kernel_sizes = np.zeros(ntrain,int)
itrain = 0
for iconf in trainrange:
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in xrange(lmax[spe]+1):
            msize = 2*l+1
            for im in xrange(msize):
                for iat in xrange(atom_counting_training[itrain,ispe]):
                    for imm in xrange(msize):
                        kernel_sizes[itrain] += 1
    itrain += 1

print "Computing regression matrices ..."

# compute regression arrays
start = time.time()
Avec,Bmat = matrices.getab(train_configs,atomic_species,llmax,nnmax,nspecies,ntrain,M,natmax,natoms_train,totsize,atomicindx_training,atom_counting_training,fps_species,almax,anmax,total_sizes,kernel_sizes) 
print "Regression matrices computed in", (time.time()-start)/60.0, "minutes"

# save regression arrays
np.save("A_vector.npy", Avec)
np.save("B_matrix.npy", Bmat)
