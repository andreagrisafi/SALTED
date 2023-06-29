import os
import numpy as np
import time
import random
from random import shuffle
from scipy import sparse
from sys_utils import read_system,get_atom_idx
import sys
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

species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut

if inp.combo:
    kdir = "kernels_"+inp.saltedname+"_"+inp.saltedname2 
    fdir = "rkhs-vectors_"+inp.saltedname+"_"+inp.saltedname2 
else:
    kdir = "kernels_"+inp.saltedname
    fdir = "rkhs-vectors_"+inp.saltedname


atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

##########################################################################################

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
atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

##########################################################################################

# compute the weight-vector size 
Mcut = {}
totsize = 0
iii=0
for spe in species:
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            Mcut[(spe,l,n)] = np.load(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
            totsize += Mcut[(spe,l,n)]
            iii+=1

print("problem dimensionality:", totsize)

dirpath = os.path.join(inp.saltedpath,fdir)
if (rank == 0):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    dirpath = os.path.join(inp.saltedpath+fdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

# Distribute structures to tasks
if inp.parallel:
    if rank == 0:
        conf_range = [[] for _ in range(size)]
        blocksize = int(round(ndata/float(size)))
        for i in range(size):
            if i == (size-1):
                conf_range[i] = list(range(ndata))[i*blocksize:ndata]
            else:
                conf_range[i] = list(range(ndata))[i*blocksize:(i+1)*blocksize]
    else:
        conf_range = None
    conf_range = comm.scatter(conf_range,root=0)
else:
    conf_range = range(ndata)

print('Task',rank+1,'handles the following structures:',conf_range,flush=True)

for iconf in conf_range:

    start = time.time()
    print(iconf,flush=True)

    # load reference QM data
    coefs = np.load(inp.saltedpath+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    Tsize = len(coefs)

    # initialize RKHS feature vectors for each channel 
    Psi = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            lsize = natoms_per_spe[(iconf,spe)]*(2*l+1) 
            for n in range(nmax[(spe,l)]):
                Psi[(spe,l,n)] = np.zeros((lsize,totsize)) 

    # fill basis set dictionary of feature vectors to be diagonal in for each channel (spe,l,n)  
    ispe = {}
    isize = 0
    iii = 0
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            psi_nm = np.load(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
            Mcut = psi_nm.shape[1]
            for n in range(nmax[(spe,l)]):
                Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
                isize += Mcut
                iii += 1

    # build sparse feature-vector memory efficiently

    nrows = Tsize
    ncols = totsize
    srows = []
    scols = []
    psi_nonzero = []
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            i1 = ispe[spe]*(2*l+1)
            i2 = ispe[spe]*(2*l+1)+2*l+1
            for n in range(nmax[(spe,l)]):
                x = Psi[(spe,l,n)][i1:i2]
                srows += list(np.nonzero(x)[0]+i) 
                scols += list(np.nonzero(x)[1])
                psi_nonzero += list(x[x!=0])
                i += 2*l+1
        ispe[spe] += 1
    ij = np.vstack((srows,scols))

    if inp.parallel:
        del srows
        del scols

    sparse_psi = sparse.coo_matrix((psi_nonzero, ij), shape=(nrows, ncols))
    sparse.save_npz(inp.saltedpath+fdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz", sparse_psi)

    if inp.parallel:
        del sparse_psi
        del psi_nonzero
        del ij
        gc.collect()

#    print(time.time()-start)
