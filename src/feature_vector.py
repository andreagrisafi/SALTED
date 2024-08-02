import os
import numpy as np
import time
from scipy import sparse
from sys_utils import read_system,get_atom_idx,get_conf_range
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

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
#start = time.time()

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut

kdir = inp.kerndir
fdir = inp.featdir

atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,spelist,atomic_symbols)

# compute the weight-vector size 
cuml_Mcut = {}
totsize = 0
for spe in spelist:
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            Mcut = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
            cuml_Mcut[(spe,l,n)] = totsize
            totsize += Mcut

print("problem dimensionality:", totsize)

dirpath = os.path.join(inp.path2ml,fdir)
if (rank == 0):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    dirpath = os.path.join(inp.path2ml+fdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

# Distribute structures to tasks
if inp.parallel:
    conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
#    if rank == 0:
#        conf_range = [[] for _ in range(size)]
#        blocksize = int(ndata/float(size))
#        for i in range(size):
#            if i == (size-1):
#                conf_range[i] = list(range(ndata))[i*blocksize:ndata]
#            else:
#                conf_range[i] = list(range(ndata))[i*blocksize:(i+1)*blocksize]
#    else:
#        conf_range = None
    conf_range = comm.scatter(conf_range,root=0)
else:
    conf_range = range(ndata)

print('Task',rank+1,'handles the following structures:',conf_range,flush=True)

for iconf in conf_range:
    print(iconf,flush=True)

    # load reference QM data
    coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    Tsize = len(coefs)

    # initialize RKHS feature vectors for each channel 
    Psi = {}

    # fill basis set dictionary of feature vectors to be diagonal in for each channel (spe,l,n)  
    ispe = {}
    for spe in spelist:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
            Psi[(spe,l)] = psi_nm

#   define a numpy equivalent to an appendable list
    class arraylist:
        def __init__(self):
            self.data = np.zeros((100000,))
            self.capacity = 100000
            self.size = 0

        def update(self, row):
            n = row.shape[0]
            self.add(row,n)

        def add(self, x, n):
            if self.size+n >= self.capacity:
                self.capacity *= 2
                newdata = np.zeros((self.capacity,))
                newdata[:self.size] = self.data[:self.size]
                self.data = newdata

            self.data[self.size:self.size+n] = x
            self.size += n

        def finalize(self):
            return self.data[:self.size]

#    srows = arraylist()
#    for l in range(lmax[spe]+1):
#        x = Psi[(spe,l)][i1:i2]
#        nz = np.nonzero(x)
#        srows.update(nz[0])
#    srows = srows.finalize()

    # build sparse feature-vector memory efficiently
    nrows = Tsize
    ncols = totsize
    srows = arraylist()
    scols = arraylist()
    psi_nonzero = arraylist()
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            i1 = ispe[spe]*(2*l+1)
            i2 = ispe[spe]*(2*l+1)+2*l+1
            x = Psi[(spe,l)][i1:i2]
            nz = np.nonzero(x)
            vals = x[x!=0]
            for n in range(nmax[(spe,l)]):
                psi_nonzero.update(vals)
                srows.update(nz[0]+i)
                scols.update(nz[1]+cuml_Mcut[(spe,l,n)])
                i += 2*l+1
        ispe[spe] += 1
    psi_nonzero = psi_nonzero.finalize()
    srows = srows.finalize()
    scols = scols.finalize()
    ij = np.vstack((srows,scols))

    if inp.parallel:
        del srows
        del scols

    sparse_psi = sparse.coo_matrix((psi_nonzero, ij), shape=(nrows, ncols))
    sparse.save_npz(inp.path2ml+fdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz", sparse_psi)

    if inp.parallel:
        del sparse_psi
        del psi_nonzero
        del ij
        gc.collect()

#    print(time.time()-start)
