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

from salted.lib import equicomb, antiequicomb, equicombnonorm, antiequicombnonorm
from salted.lib import equicombsparse, antiequicombsparse

def build():

    inp = ParseConfig().parse_input()

    # salted parameters
    (saltedname, saltedpath, saltedtype,
    filename, species, average, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        rank=0
        size=1

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    frames = read(filename,":")

    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")

    # Distribute structures to tasks
    if parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = range(ndata)

    sparse_set = np.loadtxt(osp.join(sdir, f"sparse_set_{Menv}.txt"),int)
    fps_idx = sparse_set[:,0]
    fps_species = sparse_set[:,1]

    Mspe = {}
    for spe in species:
        Mspe[spe] = 0

    fps_indexes_per_conf = {}
    for iconf in range(ndata):
        for spe in species:
            fps_indexes_per_conf[(iconf,spe)] = []

    itot = 0
    Midx_spe = {}
    for iconf in range(ndata):
        for spe in species:
            Midx_spe[(iconf,spe)] = Mspe[spe]
        for iat in range(natoms[iconf]):
            if itot in fps_idx:
                for spe in species:
                    if iat in atom_idx[(iconf,spe)]:
                        fps_indexes_per_conf[(iconf,spe)].append(iat)
                        Mspe[spe] += 1
            itot += 1

    if saltedtype=="density" or saltedtype=="ghost-density":

        # Load sparsification details if needed
        if sparsify:
            vfps = {}
            for lam in range(lmax_max+1):
                vfps[lam] = np.load(osp.join(
                    saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
                ))

        power_env_sparse = {}
        for spe in species:
            for lam in range(lmax_max+1):
                if sparsify:
                    featsize = ncut
                else:
                    [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)
                    featsize = nspe1*nspe2*nrad1*nrad2*llmax
                if lam==0:
                    power_env_sparse[(spe,lam)] = np.zeros((Mspe[spe],featsize))
                else:
                    power_env_sparse[(spe,lam)] = np.zeros((Mspe[spe],(2*lam+1),featsize))

        for iconf in conf_range:

            start_time = time.time()
            print(f"conf: {iconf+1}", flush=True)

            structure = frames[iconf]

            # Compute spherical harmonics expansion coefficients
            omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms[iconf])
            omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms[iconf])

            # Reshape arrays of expansion coefficients for optimal Fortran indexing
            v1 = np.transpose(omega1,(2,0,3,1))
            v2 = np.transpose(omega2,(2,0,3,1))

            # Compute equivariant features for the given structure
            for lam in range(lmax_max+1):

                [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

                # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
                wigner3j = np.loadtxt(os.path.join(
                    saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
                ))
                wigdim = wigner3j.size

                # Compute complex to real transformation matrix for the given lambda value
                c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

                # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
                if sparsify:

                    featsize = nspe1*nspe2*nrad1*nrad2*llmax
                    nfps = len(vfps[lam])
                    p = equicombsparse.equicombsparse(natoms[iconf],nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize,nfps,vfps[lam])
                    p = np.transpose(p,(2,0,1))
                    featsize = ncut

                else:

                    featsize = nspe1*nspe2*nrad1*nrad2*llmax
                    p = equicomb.equicomb(natoms[iconf],nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
                    p = np.transpose(p,(2,0,1))
                
                # Fill vector of equivariant descriptor
                if lam==0:
                    power = p.reshape(natoms[iconf],featsize)
                else:
                    power = p.reshape(natoms[iconf],2*lam+1,featsize)

                for spe in species:
                    nfps = len(fps_indexes_per_conf[(iconf,spe)])
                    power_env_sparse[(spe,lam)][Midx_spe[(iconf,spe)]:Midx_spe[(iconf,spe)]+nfps] = power[fps_indexes_per_conf[(iconf,spe)]]

            end_time = time.time()
            #print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)

        if parallel:
            comm.Barrier()
            for spe in species:
                for lam in range(lmax[spe]+1):
                    power_env_sparse[(spe,lam)] = comm.allreduce(power_env_sparse[(spe,lam)])

        if rank==0:
            # reshape sparse vector and save
            h5f = h5py.File(osp.join(sdir,  f"FEAT_M-{Menv}.h5"), 'w')
            for spe in species:
                for lam in range(lmax[spe]+1):
                    power_env_sparse[(spe,lam)] = power_env_sparse[(spe,lam)].reshape(Mspe[spe]*(2*lam+1),power_env_sparse[(spe,lam)].shape[-1])
                    h5f.create_dataset(f"sparse_descriptors/{spe}/{lam}",data=power_env_sparse[(spe,lam)])
            h5f.close()

    elif saltedtype=="density-response":

        lmax_max += 1
        for spe in species:
            lmax[spe] += 1

        power_env_sparse = {}
        for spe in species:
            for lam in range(lmax_max+1):
                [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)
                featsize = nspe1*nspe2*nrad1*nrad2*llmax
                if lam==0:
                    power_env_sparse[(spe,lam)] = np.zeros((Mspe[spe],featsize))
                else:
                    power_env_sparse[(spe,lam)] = np.zeros((Mspe[spe],(2*lam+1),featsize))

        for iconf in conf_range:
    
            start_time = time.time()
            print(f"conf: {iconf+1}", flush=True)
    
            structure = frames[iconf]
    
            # Compute spherical harmonics expansion coefficients
            omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms[iconf])
            omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms[iconf])
    
            # Reshape arrays of expansion coefficients for optimal Fortran indexing
            v1 = np.transpose(omega1,(2,0,3,1))
            v2 = np.transpose(omega2,(2,0,3,1))
    
            # Compute equivariant features for the given structure
            for lam in range(lmax_max+1):
    
                [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)
    
                # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
                wigner3j = np.loadtxt(os.path.join(
                    saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
                ))
                wigdim = wigner3j.size
    
                # Compute complex to real transformation matrix for the given lambda value
                c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]
    
                # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
                featsize = nspe1*nspe2*nrad1*nrad2*llmax
                p = equicombnonorm.equicombnonorm(natoms[iconf],nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
                p = np.transpose(p,(2,0,1))
    
                # Fill vector of equivariant descriptor
                if lam==0:
                    power = p.reshape(natoms[iconf],featsize)
                else:
                    power = p.reshape(natoms[iconf],2*lam+1,featsize)
    
                for spe in species:
                    nfps = len(fps_indexes_per_conf[(iconf,spe)])
                    power_env_sparse[(spe,lam)][Midx_spe[(iconf,spe)]:Midx_spe[(iconf,spe)]+nfps] = power[fps_indexes_per_conf[(iconf,spe)]]
    
            end_time = time.time()
            #print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)
    
        if parallel:
            comm.Barrier()
            for spe in species:
                for lam in range(lmax[spe]+1):
                    power_env_sparse[(spe,lam)] = comm.allreduce(power_env_sparse[(spe,lam)])
    
        if rank==0:
            # reshape sparse vector and save
            h5f = h5py.File(osp.join(sdir,  f"FEAT_M-{Menv}.h5"), 'w')
            for spe in species:
                for lam in range(lmax[spe]+1):
                    power_env_sparse[(spe,lam)] = power_env_sparse[(spe,lam)].reshape(Mspe[spe]*(2*lam+1),power_env_sparse[(spe,lam)].shape[-1])
                    h5f.create_dataset(f"sparse_descriptors/{spe}/{lam}",data=power_env_sparse[(spe,lam)])
            h5f.close()

        print("Computing antisymmetric sparse descriptors for density-response representation.")

        power_env_sparse_antisymm = {}
        for spe in species:
            for lam in range(1,lmax_max):
                [llmax,llvec] = sph_utils.get_angular_indexes_antisymmetric(lam,nang1,nang2)
                featsize = nspe1*nspe2*nrad1*nrad2*llmax
                power_env_sparse_antisymm[(spe,lam)] = np.zeros((Mspe[spe],(2*lam+1),featsize))

        for iconf in conf_range:

            structure = frames[iconf]
            print(f"conf: {iconf+1}", flush=True)

            # Compute spherical harmonics expansion coefficients
            omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms[iconf])
            omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms[iconf])

            # Reshape arrays of expansion coefficients for optimal Fortran indexing
            v1 = np.transpose(omega1,(2,0,3,1))
            v2 = np.transpose(omega2,(2,0,3,1))

            # Compute equivariant features for the given structure
            for lam in range(1,lmax_max):

                [llmax,llvec] = sph_utils.get_angular_indexes_antisymmetric(lam,nang1,nang2)

                # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
                wigner3j = np.loadtxt(os.path.join(
                    saltedpath, "wigners", f"wigner_antisymm_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
                ))
                wigdim = wigner3j.size

                # Compute complex to real transformation matrix for the given lambda value
                c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

                # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
                featsize = nspe1*nspe2*nrad1*nrad2*llmax
                p = antiequicombnonorm.antiequicombnonorm(natoms[iconf],nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
                p = np.transpose(p,(2,0,1))

                power = p.reshape(natoms[iconf],2*lam+1,featsize)

                for spe in species:
                    nfps = len(fps_indexes_per_conf[(iconf,spe)])
                    power_env_sparse_antisymm[(spe,lam)][Midx_spe[(iconf,spe)]:Midx_spe[(iconf,spe)]+nfps] = power[fps_indexes_per_conf[(iconf,spe)]]

        if parallel:
            comm.Barrier()
            for spe in species:
                for lam in range(1,lmax[spe]):
                    power_env_sparse_antisymm[(spe,lam)] = comm.allreduce(power_env_sparse_antisymm[(spe,lam)])

        if rank==0:
            # reshape sparse vector and save
            h5f = h5py.File(osp.join(sdir,  f"FEAT_M-{Menv}_antisymm.h5"), 'w')
            for spe in species:
                for lam in range(1,lmax[spe]):
                    power_env_sparse_antisymm[(spe,lam)] = power_env_sparse_antisymm[(spe,lam)].reshape(Mspe[spe]*(2*lam+1),power_env_sparse_antisymm[(spe,lam)].shape[-1])
                    h5f.create_dataset(f"sparse_descriptors/{spe}/{lam}",data=power_env_sparse_antisymm[(spe,lam)])
            h5f.close()

        end_time = time.time()
        #print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)



if __name__ == "__main__":
    build()
