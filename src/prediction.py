#!/usr/bin/python

import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle
import argparse
import prediction

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-s",   "--trainset",     type=int,   default=1,   help="training dataset selection")
    parser.add_argument("-f",   "--trainfrac"  , type=float, default=1.0, help="training set fraction")
    parser.add_argument("-m",   "--msize"  ,     type=int,   default=100, help="number of reference environments")
    parser.add_argument("-rc",  "--cutoffradius"  , type=float, default=4.0, help="soap cutoff")
    parser.add_argument("-sg",  "--sigmasoap"  , type=float, default=0.3, help="soap sigma")
    parser.add_argument("-r",   "--regular"  , type=float, default=1e-06, help="regularization")
    parser.add_argument("-jit", "--jitter"  , type=float, default=1e-10, help="jitter")
    parser.add_argument("-ts", "--testset", type=str,required=False,help="-full- if the full test set need to be selected")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    s = args.trainset
    f = args.trainfrac
    m = args.msize
    rc = args.cutoffradius
    sg = args.sigmasoap
    r = args.regular
    jit = args.jitter
    ts = args.testset
    return [s,f,m,rc,sg,r,jit,ts]

args = add_command_line_arguments_contraction("density regression")
[nset,frac,M,rc,sigma_soap,reg,jit,tset] = set_variable_values_contraction(args)

# coversion factors
bohr2ang = 0.529177249

mol = "water"

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

# dataset partitioning
trainrangetot = np.loadtxt("SELECTIONS/training_selection.txt",int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]
print "Number of training molecules = ", ntrain
print "Number of testing molecules = ", ntest

# define testing indexes 
test_configs = np.array(testrange,int)
atomicindx_test = atomicindx[:,:,testrange]
atom_counting_test = atom_counting[testrange]
test_species = np.zeros((ntest,natmax),int)
for itest in xrange(ntest):
    for iat in xrange(natoms_test[itest]):
        test_species[itest,iat] = spec_list_per_conf[testrange[itest]][iat]

# sparse kernel sizes 
kernel_sizes = np.zeros(ntest,int)
itest = 0
for iconf in testrange:
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in xrange(lmax[spe]+1):
            msize = 2*l+1
            for im in xrange(msize):
                for iat in xrange(atom_counting_test[itest,ispe]):
                    for imm in xrange(msize):
                        kernel_sizes[itest] += 1
    itest += 1

# load regression weights 
weights = np.load("WEIGHTS/weights_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

# unravel regression weights with explicit indexing
ww = np.zeros((M,llmax+1,nnmax,2*llmax+1),float)
i = 0
for ienv in xrange(M):
    ispe = fps_species[ienv]
    al = almax[ispe]
    for l in xrange(al):
        msize = 2*l+1
        anc = anmax[ispe,l]
        for n in xrange(anc):
            for im in xrange(msize):
                ww[ienv,l,n,im] = weights[i] 
                i += 1

# load testing kernels and perform prediction 
coeffs = prediction.prediction(mol,kernel_sizes,fps_species,atom_counting_test,atomicindx_test,nspecies,ntest,int(rc),natmax,llmax,nnmax,natoms_test,test_configs,test_species,almax,anmax,M,ww)

np.save("PREDICTIONS/prediction_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",coeffs)

