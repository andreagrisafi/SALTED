import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle
from scipy import sparse

from sys_utils import read_system,get_atom_idx
import basis

sys.path.insert(0, './')
import inp

if inp.parallel:
    import gc
    from mpi4py import MPI
    # MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size)
else:
    rank=0

if inp.combo:
    fdir = "rkhs-vectors_"+inp.saltedname+"_"+inp.saltedname2
    rdir = "regrdir_"+inp.saltedname+"_"+inp.saltedname2
else:
    fdir = "rkhs-vectors_"+inp.saltedname
    rdir = "regrdir_"+inp.saltedname

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
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

coefdir = inp.coefdir

species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)


###############################################################################################

natoms_total = 0
natoms_list = []
natoms = np.zeros(ndata,int)
for iconf in range(ndata):
    natoms[iconf] = 0
    for spe in species:
        natoms[iconf] += natoms_per_spe[(iconf,spe)]
    natoms_total += natoms[iconf]
    natoms_list.append(natoms[iconf])
    # Define excluded species
    excluded_species = []
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))
natmax = max(natoms_list)

# recompute atomic indexes from new species selections
atoms_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

###############################################################################################

p = sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf0.npz")
totsize = p.shape[-1]
print("problem dimensionality:", totsize,flush=True)
if totsize>70000:
    print("ERROR: problem dimension too large, minimize directly loss-function instead!")
    sys.exit(0)

if inp.average:
    # load average density coefficients
    av_coefs = {}
    for spe in spelist:
        av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

if rank==0:
    dirpath = os.path.join(inp.saltedpath, rdir)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    dirpath = os.path.join(inp.saltedpath+rdir+"/", "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

# define training set at random
dataset = list(range(ndata))
#random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt(inp.saltedpath+rdir+"/training_set_N"+str(inp.Ntrain)+".txt",trainrangetot,fmt='%i')
ntraintot = int(inp.trainfrac*inp.Ntrain)

# Distribute structures to tasks
if inp.parallel:
    if rank == 0 and ntraintot < size:
        print('You have requested more processes than training structures. Please reduce the number of processes',flush=True)
        comm.Abort()
    if rank == 0:
        trainrange = [[] for _ in range(size)]
        blocksize = int(round(ntraintot/float(size)))
        for i in range(size):
            if i == (size-1):
                trainrange[i] = trainrangetot[i*blocksize:ntraintot]
            else:
                trainrange[i] = trainrangetot[i*blocksize:(i+1)*blocksize]
    else:
        trainrange = None

    trainrange = comm.scatter(trainrange,root=0)
    print('Task',rank+1,'handles the following structures:',trainrange,flush=True)
else:
    trainrange = trainrangetot[:ntraintot]
ntrain = len(trainrange)

if rank==0:
    print("computing regression matrices...")
#Avec = np.load(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N240.npy")*240
#Bmat = np.load(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N240.npy")*240
Avec = np.zeros(totsize)
Bmat = np.zeros((totsize,totsize))
for iconf in trainrange:
    print("conf:", iconf+1,flush=True)
   
    # load reference QM data
    ref_coefs = np.load(inp.saltedpath+coefdir+"coefficients_conf"+str(iconf)+".npy")
    over = np.load(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy")
    psivec = sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz")
    psi = psivec.toarray()

    if inp.average:

        # fill array of average spherical components
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            if spe in species:
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if l==0:
                           Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1
        
        # subtract average
        ref_coefs -= Av_coeffs
    
    ref_projs = np.dot(over,ref_coefs)
    
    Avec += np.dot(psi.T,ref_projs)
    Bmat += np.dot(psi.T,np.dot(over,psi))

    if iconf+1==160 or iconf+1==200 or iconf+1==220: 
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(iconf+1)+".npy",Avec/float(iconf+1))
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(iconf+1)+".npy",Bmat/float(iconf+1))

if inp.parallel:
    Avec = comm.allreduce(Avec) / float(ntrain)
    Bmat = comm.allreduce(Bmat) / float(ntrain)
else:
    Avec /= float(ntrain)
    Bmat /= float(ntrain)

if rank==0:
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(ntrain)+".npy",Avec)
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(ntrain)+".npy",Bmat)
