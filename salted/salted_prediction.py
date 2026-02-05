import os
import sys
import time
import h5py
import numpy as np
from scipy import special
from ase.data import atomic_numbers
from ase.io import read

from salted.lib import equicomb 
from salted.lib import equicombsparse

from salted import sph_utils
from salted import basis
from salted.sys_utils import ParseConfig, get_conf_range

def build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,comm,size,rank,structure):

    inp = ParseConfig().parse_input()

    (saltedname, saltedpath, saltedtype,
    filename, species, average, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    # read system
    ndata = len(structure)
    
    # Define system excluding atoms that belong to species not listed in SALTED input 
    atomic_symbols = structure.get_chemical_symbols()
    natoms_tot = len(atomic_symbols)
    excluded_species = []
    for iat in range(natoms_tot):
        spe = atomic_symbols[iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols = list(filter(lambda a: a != spe, atomic_symbols))
    natoms = int(len(atomic_symbols))
  
    if parallel:

        if natoms < size:
            if rank == 0:
                raise ValueError(
                    f"More processes {size=} have been requested than atoms {natoms=}. "
                    f"Please reduce the number of processes."
                )
            else:
                exit()
        atoms_range = get_conf_range(rank, size, natoms, np.arange(natoms,dtype=int))
        atoms_range = comm.scatter(atoms_range, root=0)
        print(
            f"Task {rank+1} handles the following atoms: {atoms_range}", flush=True
        )

    else:

        atoms_range = np.arange(natoms,dtype=int)

    natoms_range = int(len(atoms_range))
    atomic_symbols_range = [atomic_symbols[i] for i in atoms_range]
 
    atom_idx = {}
    natom_dict = {}
    for spe in species:
        atom_idx[spe] = []
        natom_dict[spe] = 0
    for iat in range(natoms_range):
        spe = atomic_symbols_range[iat]
        if spe in species:
           atom_idx[spe].append(iat)
           natom_dict[spe] += 1

    pseudocharge = inp.qm.pseudocharge
    pseudocharge_dict = {}
    for i in range(len(species)):
        pseudocharge_dict[species[i]] = pseudocharge[i] # Warning: species and pseudocharge must have the same ordering
    
    omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe1,species,nang1,nrad1,natoms)
    omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe2,species,nang2,nrad2,natoms)

    # Retain only the atoms of the given MPI task
    omega1 = omega1[:,atoms_range]
    omega2 = omega2[:,atoms_range]

    # Reshape arrays of expansion coefficients for optimal Fortran indexing 
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    pvec = {}
    for lam in range(lmax_max+1):
    
#        print("lambda =", lam)
    
        equistart = time.time()
    
        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)
 
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
            p = equicombsparse.equicombsparse(natoms_range,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize,nfps,vfps[lam])
            p = np.transpose(p,(2,0,1))
            featsize = ncut

        else:

            featsize = nspe1*nspe2*nrad1*nrad2*llmax
            p = equicomb.equicomb(natoms_range,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
            p = np.transpose(p,(2,0,1))

        if lam==0: 
            pvec[lam] = p.reshape(natoms_range,featsize)
        else:
            pvec[lam] = p.reshape(natoms_range,2*lam+1,featsize)
        
        # print("equicomb time:", (time.time()-equistart))
    
    rkhsstart = time.time()
 
    psi_nm = {}
    for spe in species:

        # lam = 0
        if zeta==1:
            psi_nm[(spe,0)] = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
        else:
            kernel0_nm = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
            kernel_nm = kernel0_nm**zeta
            psi_nm[(spe,0)] = np.dot(kernel_nm,Vmat[(0,spe)])

        # lam > 0
        for lam in range(1,lmax[spe]+1):

            featsize = pvec[lam].shape[-1]
            if zeta==1:
                psi_nm[(spe,lam)] = np.dot(pvec[lam][atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
            else:
                kernel_nm = np.dot(pvec[lam][atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                kernel_nm_blocks = kernel_nm.reshape(natom_dict[spe], 2*lam+1, Mspe[spe], 2*lam+1)
                kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (zeta - 1)
                kernel_nm = kernel_nm_blocks.reshape(natom_dict[spe]*(2*lam+1), Mspe[spe]*(2*lam+1))
                #for i1 in range(natom_dict[spe]):
                #    for i2 in range(Mspe[spe]):
                #        kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[i1,i2]**(zeta-1)
                psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])

    #if print("rkhs time:", time.time()-rkhsstart,flush=True)
 
    # Perform equivariant predictions
    predstart = time.time()
    
    # Load spherical averages if required
    if average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))
    
    Tsize = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Tsize += 2*l+1
    
    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = psi_nm[(spe,l)].shape[1]
                C[(spe,l,n)] = np.dot(psi_nm[(spe,l)],weights[isize:isize+Mcut])
                isize += Mcut
    
    # init averages array if asked
    if average:
        Av_coeffs = np.zeros(Tsize)

    # fill vector of predictions
    itot = 0
    pred_coefs = np.zeros(Tsize)
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if iat in atoms_range:
            i=0
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    pred_coefs[itot+i:itot+i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                    i += 2*l+1
            ispe[spe] += 1
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                if average and l==0:
                    Av_coeffs[itot] = av_coefs[spe][n]
                itot += 2*l+1

    # add back spherical averages if required
    if average and rank==0:
        pred_coefs += Av_coeffs

    if parallel:
        comm.Barrier()
        pred_coefs = comm.allreduce(pred_coefs)

    if qmcode=="cp2k" and rank==0:

        charge, dipole = compute_charge_and_dipole(structure,inp.qm.pseudocharge,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs,average)
        print(iconf+1,charge,dipole["x"],dipole["y"],dipole["z"],flush=True) 

#    if print("pred time:", time.time()-predstart,flush=True)
    
    return pred_coefs

if __name__ == "__main__":
    build()
