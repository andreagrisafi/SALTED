import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse
import matrices 

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-s",   "--testset",     type=int,   default=1,   help="test dataset selection")
    parser.add_argument("-f",   "--trainfrac"  , type=float, default=1.0, help="training set fraction")
    parser.add_argument("-m",   "--msize"  ,     type=int,   default=100, help="number of reference environments")
    parser.add_argument("-rc",   "--cutoffradius"  , type=float, default=4.0, help="soap cutoff")
    parser.add_argument("-sg",   "--sigmasoap"  , type=float, default=0.3, help="soap sigma")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    s = args.testset
    f = args.trainfrac
    m = args.msize
    rc = args.cutoffradius
    sg = args.sigmasoap
    return [s,f,m,rc,sg]

args = add_command_line_arguments_contraction("density regression")
[nset,frac,M,rc,sigma_soap] = set_variable_values_contraction(args)

# conversion constants
bohr2ang = 0.529177249

# system definition
mol = "water"
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
print "problem dimensionality =", totsize

# training set selection 
trainrangetot = np.loadtxt("SELECTIONS/training_selection.txt",int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
print "Number of training molecules = ", ntrain

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

# compute regression arrays
start = time.time()
Avec,Bmat = matrices.getab(mol,train_configs,atomic_species,llmax,nnmax,nspecies,ntrain,M,natmax,natoms_train,int(rc),totsize,atomicindx_training,atom_counting_training,fps_species,almax,anmax,total_sizes,kernel_sizes) 
print "A-vector and B-matrix computed in", time.time()-start, "seconds"

# save regression arrays
np.save("MATRICES/Avec_M"+str(M)+"_trainfrac"+str(frac)+".npy", Avec)
np.save("MATRICES/Bmat_M"+str(M)+"_trainfrac"+str(frac)+".npy", Bmat)
