import sys
import os
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

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-p", "--partial", type=int, default=0, help="Calculate A and B for the pth block of structures in the training set")
    parser.add_argument("-b", "--partial_blocks", type=int, default=10, help="The number of structures included in each block, when using --partial")

    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("density regression")
# read partial matrix information
p = args.partial
b = args.partial_blocks

# Is automatic cross-validation requested?
xv = inp.xv
if xv and (p > 0): sys.exit('The options xv and --partial are not compatible')

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

# number of training configurations 
N = inp.Ntrain

# training set fraction
frac = inp.trainfrac

# when performing an internal cross-validation,
# Ntrain and frac are set to include half of the dataset
if xv:
    print "Automatic cross-validation requested. Ntrain and frac set automatically"
    N = ndata//2
    frac = 1.0

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
validate_range = np.setdiff1d(dataset,trainrange).tolist()

dirpath = os.path.join(inp.path2ml, "matrices")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

for iloop in range(2):
    # Two loops only performed when cross-validating
    if iloop == 1 and not xv: continue

    if iloop == 1 : trainrange = validate_range
    natoms_train = natoms[trainrange]
    print "Number of training configurations =", ntrain

    # Include only the pth block of b training structures
    if p > 0:
        print 'Calculating the ',p,'th block of ', b,' configurations'
        ntrain_tot = ntrain
        if b*(p-1) >= ntrain_tot: sys.exit('The requested block contains no structures. Reduce -p or -b')
        ntrain = b
        if b*p > ntrain_tot:
            ntrain = ntrain_tot - b*(p-1)
            print 'WARNING: The ',p,'th block contains only ',ntrain,' structures, fewer than the requested block size, because the end of the dataset was reached'
            trainrange = trainrange[b*(p-1):ntrain_tot]
        else:
            trainrange = trainrange[b*(p-1):b*p]
        natoms_train = natoms[trainrange]
    
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

    path2kerns = inp.path2ml+"kernels/"
    path2overl = inp.path2qm+"overlaps/"
    path2projs = inp.path2qm+"projections/"

    # compute regression arrays
    start = time.time()
    Avec,Bmat = matrices.getab(path2kerns,path2overl,path2projs,train_configs,atomic_species,llmax,nnmax,nspecies,ntrain,M,natmax,natoms_train,totsize,atomicindx_training,atom_counting_training,fps_species,almax,anmax,total_sizes,kernel_sizes) 
    print "Regression matrices computed in", (time.time()-start)/60.0, "minutes"

    # save regression arrays
    if iloop == 0:
       if p > 0:
           np.save(inp.path2ml+"matrices/A_"+str(p)+"_vector.npy", Avec)
           np.save(inp.path2ml+"matrices/B_"+str(p)+"_matrix.npy", Bmat)
       else:
           np.save(inp.path2ml+"matrices/A_vector.npy", Avec)
           np.save(inp.path2ml+"matrices/B_matrix.npy", Bmat)
    if iloop == 1:
        np.save(inp.path2ml+"matrices/Ap_vector.npy", Avec)
        np.save(inp.path2ml+"matrices/Bp_matrix.npy", Bmat)
