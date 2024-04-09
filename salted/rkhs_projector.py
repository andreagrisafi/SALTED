"""
TODO: replace class arraylist by numpy.concatenate
"""

import os
import sys
import time
import os.path as osp
from ase.io import read
import h5py

import numpy as np
from scipy import sparse
from ase.data import atomic_numbers

from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

from salted import wigner
from salted import sph_utils
from salted import basis

from salted.lib import equicomb


def build():
    inp = ParseConfig().parse_input()

    # salted parameters
    (saltedname, saltedpath,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()

    if parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank=0
        size=1

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")

    # compute rkhs projector and save
    features = h5py.File(osp.join(sdir,f"FEAT_M-{M}.h5"),'r')
    h5f = h5py.File(osp.join(sdir,  f"projector_M{M}_zeta{zeta}.h5"), 'w')
    for spe in species:
        power_env_sparse = features['sparse_descriptor'][spe]['0'][:]
        Mspe = power_env_sparse.shape[0]
        kernel0_mm = np.dot(power_env_sparse,power_env_sparse.T)
        eva, eve = np.linalg.eigh(kernel0_mm**zeta)
        print(eva)
        print(eva>eigcut)
        eva = eva[eva>eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
        h5f.create_dataset(f"projectors/{spe}/0",data=V)
        for lam in range(1,lmax[spe]+1):
            power_env_sparse = features['sparse_descriptor'][spe][str(lam)][:]
            kernel_mm = np.dot(power_env_sparse,power_env_sparse.T)
            for i1 in range(Mspe):
                for i2 in range(Mspe):
                    kernel_mm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_mm[i1,i2]**(zeta-1)
            eva, eve = np.linalg.eigh(kernel_mm)
            eva = eva[eva>eigcut]
            eve = eve[:,-len(eva):]
            V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
            h5f.create_dataset(f"projectors/{spe}/{lam}",data=V)
    h5f.close()
<<<<<<< HEAD
    features.close()

=======
>>>>>>> 51868a9 (Store sparse features and projectors in single files)

if __name__ == "__main__":
    build()
