"""
TODO: replace class arraylist by numpy.concatenate
"""

import os
import sys
import time
import os.path as osp

import numpy as np
from scipy import sparse

from salted.sys_utils import read_system,get_atom_idx,get_conf_range

def build():

    sys.path.insert(0, './')
    import inp

    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank=0
        size=1

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

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

    # sparse-GPR parameters
    M = inp.Menv
    zeta = inp.z

    if inp.field:
        kdir = f"kernels_{inp.saltedname}_field"
        fdir = f"rkhs-vectors_{inp.saltedname}_field"
    else:
        kdir = f"kernels_{inp.saltedname}"
        fdir = f"rkhs-vectors_{inp.saltedname}"


    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # compute the weight-vector size
    cuml_Mcut = {}
    totsize = 0
    for spe in species:
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = np.load(os.path.join(
                    inp.saltedpath, kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}", f"psi-nm_conf{0}.npy"
                )).shape[1]
                cuml_Mcut[(spe,l,n)] = totsize
                totsize += Mcut

    if rank == 0: print(f"problem dimensionality: {totsize}", flush=True)

    if (rank == 0):
        dirpath = os.path.join(inp.saltedpath, fdir, f"M{M}_zeta{zeta}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    if size > 1:  comm.Barrier()

    # Distribute structures to tasks
    if inp.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = range(ndata)

    for iconf in conf_range:

        start_time = time.time()
        print(f"{iconf} start", flush=True)

        # load reference QM data
        coefs = np.load(osp.join(inp.saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"))
        Tsize = len(coefs)

        # initialize RKHS feature vectors for each channel
        Psi = {}

        ispe = {}
        for spe in species:
            ispe[spe] = 0
            for l in range(lmax[spe]+1):
                psi_nm = np.load(osp.join(
                    inp.saltedpath, kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}", f"psi-nm_conf{iconf}.npy"
                ))
                Psi[(spe,l)] = psi_nm

    # This would be needed if psi_nm depended on n
    #            Mcut = psi_nm.shape[1]
    #            for n in range(nmax[(spe,l)]):
    #                Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
    #                isize += Mcut


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
    # If psi_nm depended on n, would need the follwing lines
    #                x = Psi[(spe,l,n)][i1:i2]
    #                nz = np.nonzero(x)
    #                vals = x[x!=0]
                    psi_nonzero.update(vals)
                    srows.update(nz[0]+i)
                    scols.update(nz[1]+cuml_Mcut[(spe,l,n)])
                    i += 2*l+1
            ispe[spe] += 1

        psi_nonzero = psi_nonzero.finalize()
        srows = srows.finalize()
        scols = scols.finalize()
        ij = np.vstack((srows,scols))

        del srows
        del scols

        sparse_psi = sparse.coo_matrix((psi_nonzero, ij), shape=(nrows, ncols))
        sparse.save_npz(osp.join(
            inp.saltedpath, fdir, f"M{M}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
        ), sparse_psi)

        del sparse_psi
        del psi_nonzero
        del ij

        end_time = time.time()
        print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)



if __name__ == "__main__":
    build()
