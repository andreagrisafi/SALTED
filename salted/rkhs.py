import os
import sys
import time
import os.path as osp

import numpy as np
import h5py

from salted.sys_utils import read_system, get_atom_idx, get_conf_range

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
        rank = 0
        size = 1

    saltedname = inp.saltedname

    if inp.field:
        kdir = osp.join(inp.saltedpath, f"kernels_{inp.saltedname}_field")
    else:
        kdir = osp.join(inp.saltedpath, f"kernels_{inp.saltedname}")

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # number of sparse environments
    M = inp.Menv
    zeta = inp.z
    eigcut = inp.eigcut
    if rank == 0:
        print("M =", M, "eigcut =", eigcut)
        print("zeta =", zeta)
        print("Computing RKHS of symmetry-adapted sparse kernel approximations...")
    sdir = osp.join(inp.saltedpath, f"equirepr_{inp.saltedname}")

    # Distribute structures to tasks
    if inp.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print(f"Task {rank+1} handles the following structures: {conf_range}", flush=True)
    else:
        conf_range = list(range(ndata))

    power_env_sparse = {}
    if inp.field: power_env_sparse2 = {}
    kernel0_nm = {}

    power = h5py.File(osp.join(sdir, "FEAT-0.h5"), 'r')["descriptor"][conf_range, :]
    nfeat = power.shape[-1]
    if inp.field:
        power2 = h5py.File(osp.join(sdir, "FEAT-0_field.h5"), 'r')["descriptor"][conf_range, :]
        nfeat2 = power2.shape[-1]
    Mspe = {}

    for spe in species:
        if rank == 0: print(f"lambda = 0 species: {spe}")
        start = time.time()

        # compute sparse kernel K_MM for each atomic species
        power_env_sparse[spe] = h5py.File(osp.join(sdir, f"FEAT-0-M-{M}.h5"), 'r')[spe][:]
        if inp.field: power_env_sparse2[spe] = h5py.File(osp.join(sdir, f"FEAT-0-M-{M}_field.h5"), 'r')[spe][:]
        Mspe[spe] = power_env_sparse[spe].shape[0]

        V = np.load(osp.join(kdir, f"spe{spe}_l{0}", f"M{M}_zeta{zeta}", "projector.npy"))

        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for i,iconf in enumerate(conf_range):
            kernel0_nm[(iconf,spe)] = np.dot(power[i,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
            if inp.field:
                kernel_nm = np.dot(power2[i,atom_idx[(iconf,spe)]],power_env_sparse2[spe].T)
                kernel_nm *= kernel0_nm[(iconf,spe)]**(zeta-1)
                #kernel_nm *= np.exp(kernel0_nm[(iconf,spe)])
            else:
                kernel_nm = kernel0_nm[(iconf,spe)]**zeta
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(osp.join(
                kdir, f"spe{spe}_l{0}", f"M{M}_zeta{zeta}", f"psi-nm_conf{iconf}.npy"
            ), psi_nm)
        if rank == 0: print((time.time()-start)/60.0)

    # lambda>0
    for l in range(1,llmax+1):

        # load power spectrum
        if rank == 0:  print(f"loading lambda = {l}")
        power = h5py.File(osp.join(sdir, f"FEAT-{l}.h5"), 'r')["descriptor"][conf_range, :]
        nfeat = power.shape[-1]
        if inp.field:
            power2 = h5py.File(osp.join(sdir, f"FEAT-{l}_field.h5"), 'r')["descriptor"][conf_range, :]
            nfeat2 = power2.shape[-1]

        for spe in species:
            if rank == 0: print(f"lambda = {l} species: {spe}")
            start = time.time()

            # get sparse feature vector for each atomic species
            power_env_sparse[spe] = h5py.File(osp.join(sdir, f"FEAT-{l}-M-{M}.h5"), 'r')[spe][:]
            if inp.field: power_env_sparse2[spe] = h5py.File(osp.join(sdir, f"FEAT-{l}-M-{M}_field.h5"), 'r')[spe][:]
            V = np.load(osp.join(kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}", "projector.npy"))

            # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
            if zeta == 1:
                for i,iconf in enumerate(conf_range):
                    if inp.field:
                        kernel_nm = np.dot(power2[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat2),power_env_sparse2[spe].T)
                    else:
                        kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T)
                    psi_nm = np.real(np.dot(kernel_nm,V))
                    np.save(osp.join(
                        kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}", f"psi-nm_conf{iconf}.npy"
                    ), psi_nm)
            else:
                for i,iconf in enumerate(conf_range):
                    if inp.field:
                        kernel_nm = np.dot(power2[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat2),power_env_sparse2[spe].T)
                    else:
                        kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T)

                    for i1 in range(natom_dict[(iconf,spe)]):
                        for i2 in range(Mspe[spe]):
                            kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
                            #kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= np.exp(kernel0_nm[(iconf,spe)][i1,i2])
                    psi_nm = np.real(np.dot(kernel_nm,V))
                    np.save(osp.join(
                        kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}", f"psi-nm_conf{iconf}.npy"
                    ), psi_nm)
            if rank == 0: print(f"time cost: {((time.time()-start)):.2f} s")


if __name__ == "__main__":
    build()
