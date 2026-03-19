import os
import sys
import time

import h5py
import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from scipy import special

from salted import basis, sph_utils
from salted.cp2k.utils import compute_charge_and_dipole, scale_grad_coefs
from salted.sys_utils import ParseConfig, check_MPI_tasks_count, distribute_jobs, format_index_ranges 

def build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,size,rank,lcut,gradient,structure):

    inp = ParseConfig().parse_input()

    (saltedname, saltedpath, saltedtype,
    filename, species, average,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    start_time = time.time()

    # read system
    ndata = len(structure)
    
    # Define system excluding atoms that belong to species not listed in SALTED input 
    atomic_symbols = structure.get_chemical_symbols()
    natoms_tot = len(atomic_symbols)
    excluded_species = []
    atomic_global_idx = []
    for iat in range(natoms_tot):
        spe = atomic_symbols[iat]
        if spe not in species:
            excluded_species.append(spe)
        else:
            atomic_global_idx.append(iat)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols = list(filter(lambda a: a != spe, atomic_symbols))
    natoms = int(len(atomic_symbols))
    atomic_global_idx = np.array(atomic_global_idx,int)

    parallel = (size > 1)
    if parallel:
        check_MPI_tasks_count(comm, natoms, "atoms")
        atoms_range = distribute_jobs(comm, np.arange(natoms,dtype=int))
        if inp.salted.verbose:
            print(f"Task {rank} handles the following atoms: {format_index_ranges(atoms_range,True)}", flush=True)
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
    
    if gradient:
    
        omega1, domega1 = sph_utils.get_representation_gradient_coeffs_atomrange(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms_tot,atomic_global_idx[atoms_range])
        omega2, domega2 = sph_utils.get_representation_gradient_coeffs_atomrange(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms_tot,atomic_global_idx[atoms_range])
    
        dv1 = np.transpose(domega1.reshape((domega1.shape[0],natoms_range,natoms_tot,3,domega1.shape[3],domega1.shape[4])),(1,5,0,4,2,3)).copy()
        dv2 = np.transpose(domega2.reshape((domega2.shape[0],natoms_range,natoms_tot,3,domega2.shape[3],domega2.shape[4])),(1,5,0,4,2,3)).copy()
        
        grad_pvec = {}
   
    else: 

        omega1 = sph_utils.get_representation_coeffs_atomrange(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,atomic_global_idx[atoms_range])
        omega2 = sph_utils.get_representation_coeffs_atomrange(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,atomic_global_idx[atoms_range])
   
    v1 = np.transpose(omega1,(1,3,0,2)).copy()
    v2 = np.transpose(omega2,(1,3,0,2)).copy()
        
 
    #print("Time get rep coeffs", str(time.time()-time_omega))

    #start = time.time()

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    pvec = {}
    for lam in range(min(lmax_max,lcut)+1):
    
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

            if gradient:
      
                p, grad_p = sph_utils.grad_equicombsparse_numba(natoms_tot,natoms_range,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,dv1,dv2,wigner3j,llmax,llvec,lam,c2r,featsize,nfps,vfps[lam])
            
            else:

                p = sph_utils.equicombsparse_numba(natoms_range,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigner3j,llmax,llvec,lam,c2r,featsize,nfps,vfps[lam])

            featsize = ncut

        else:

            featsize = nspe1*nspe2*nrad1*nrad2*llmax

            if gradient:

                p, grad_p = sph_utils.grad_equicomb_numba(natoms_tot,natoms_range,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,dv1,dv2,wigner3j,llmax,llvec,lam,c2r,featsize)

            else:

                p = sph_utils.equicomb_numba(natoms_range,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigner3j,llmax,llvec,lam,c2r,featsize)

        if lam==0: 

            pvec[lam] = p.reshape(natoms_range,featsize)

            if gradient:

                grad_pvec[lam] = grad_p.reshape(natoms_tot, 3, natoms_range, featsize)

        else:

            pvec[lam] = p.reshape(natoms_range,2*lam+1,featsize)

            if gradient:
 
                grad_pvec[lam] = grad_p.reshape(natoms_tot,3,natoms_range,2*lam+1,featsize)

        #print("equicomb time:", (time.time()-equistart))
    
    rkhsstart = time.time()

    psi_nm = {}
    if gradient: grad_psi_nm = {}

    for spe in species:

        # lam = 0
        featsize = pvec[0].shape[-1]
        if zeta==1:
            psi_nm[(spe,0)] = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
            if gradient: 
                grad_psi_nm[(spe,0)] = np.dot(grad_pvec[0][:,:,atom_idx[spe],:].reshape(natoms_tot*3*natom_dict[spe],featsize),power_env_sparse[(0,spe)].T)
        else:
            kernel0_nm = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
            kernel_nm = kernel0_nm**zeta
            psi_nm[(spe,0)] = np.dot(kernel_nm,Vmat[(0,spe)])
            if gradient:
                grad_kernel0_nm = np.dot(grad_pvec[0][:,:,atom_idx[spe],:].reshape((natoms_tot*3*natom_dict[spe],featsize)),power_env_sparse[(0,spe)].T)
                grad_kernel_nm = (zeta*grad_kernel0_nm.reshape((natoms_tot,3,natom_dict[spe],Mspe[spe]))*kernel0_nm[np.newaxis, np.newaxis, :, :]**(zeta-1)).reshape((natoms_tot*3*natom_dict[spe],Mspe[spe]))
                grad_psi_nm[(spe,0)] = np.dot(grad_kernel_nm,Vmat[(0,spe)])

        # lam > 0
        for lam in range(1,min(lmax[spe],lcut)+1):

            featsize = pvec[lam].shape[-1]
            if zeta==1:
                psi_nm[(spe,lam)] = np.dot(pvec[lam][atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                if gradient: 
                    grad_psi_nm[(spe,lam)] = np.dot(grad_pvec[lam][:,:,atom_idx[spe],:,:].reshape(natoms_tot*3*natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T) 
            else:
                kernel_nm = np.dot(pvec[lam][atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                kernel_nm_blocks = kernel_nm.reshape(natom_dict[spe], 2*lam+1, Mspe[spe], 2*lam+1).copy()
                kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (zeta - 1)
                if gradient:
                    grad_kernel_nm = np.dot(grad_pvec[lam][:,:,atom_idx[spe],:,:].reshape(natoms_tot*3*natom_dict[spe]*(2*lam+1), featsize), power_env_sparse[(lam,spe)].T)
                    grad_kernel_nm_blocks = grad_kernel_nm.reshape(natoms_tot, 3, natom_dict[spe], 2*lam+1, Mspe[spe], 2*lam+1)
                    grad_kernel_nm_blocks = (grad_kernel_nm_blocks * (kernel0_nm ** (zeta - 1))[np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis]) + kernel_nm.reshape(natom_dict[spe], 2*lam+1, Mspe[spe], 2*lam+1)[np.newaxis, np.newaxis, :, :, :, :] * ((zeta-1) * (grad_kernel0_nm.reshape(natoms_tot,3,natom_dict[spe],Mspe[spe]) * (kernel0_nm ** (zeta-2))[np.newaxis, np.newaxis, :, :]))[:, :, :, np.newaxis, :, np.newaxis]
                kernel_nm = kernel_nm_blocks.reshape(natom_dict[spe]*(2*lam+1), Mspe[spe]*(2*lam+1))
                psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])
                if gradient:
                    grad_kernel_nm = grad_kernel_nm_blocks.reshape(natoms_tot*3*natom_dict[spe]*(2*lam+1), Mspe[spe]*(2*lam+1))
                    grad_psi_nm[(spe,lam)] = np.dot(grad_kernel_nm,Vmat[(lam,spe)])
                     
    #print("rkhs time:", time.time()-rkhsstart,flush=True)
 
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
        for l in range(min(lmax[spe],lcut)+1):
            for n in range(nmax[(spe,l)]):
                Tsize += 2*l+1
    
    # compute predictions per channel
    C = {}
    if gradient: grad_C = {}
    ispe = {}
    isize = 0
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = Vmat[(l,spe)].shape[1]
                if l <= lcut:
                    C[(spe,l,n)] = np.dot(psi_nm[(spe,l)],weights[isize:isize+Mcut])
                    if gradient: 
                        grad_C[(spe,l,n)] = np.dot(grad_psi_nm[(spe,l)],weights[isize:isize+Mcut])
                isize += Mcut
    
    # init averages array if asked
    if average:
        Av_coeffs = np.zeros(Tsize)

    # fill vector of predictions
    atoms_range_set = set(atoms_range)
    itot = 0
    pred_coefs = np.zeros(Tsize)
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if iat in atoms_range_set:
            i=0
            for l in range(min(lmax[spe],lcut)+1):
                for n in range(nmax[(spe,l)]):
                    pred_coefs[itot+i:itot+i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                    i += 2*l+1
            ispe[spe] += 1
        for l in range(min(lmax[spe],lcut)+1):
            for n in range(nmax[(spe,l)]):
                if average and l==0:
                    Av_coeffs[itot] = av_coefs[spe][n]
                itot += 2*l+1
    
    if gradient:

        ispe = {}
        for spe in species:
            ispe[spe] = 0
        itot = 0
        grad_pred_coefs = np.zeros((natoms_tot,3,Tsize))
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            if iat in atoms_range_set:
                i=0
                for l in range(min(lmax[spe],lcut)+1):
                    for n in range(nmax[(spe,l)]):
                        grad_pred_coefs[:,:,itot+i:itot+i+(2*l+1)] = grad_C[(spe,l,n)].reshape(natoms_tot,3,natom_dict[spe]*(2*l+1))[:,:,ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+(2*l+1)]
                        i += (2*l+1)
                ispe[spe] += 1
            for l in range(min(lmax[spe],lcut)+1):
                for n in range(nmax[(spe,l)]):
                    itot += (2*l+1)


    # add back spherical averages if required
    if average and rank==0:
        pred_coefs += Av_coeffs
    
    if parallel:
        comm.Barrier()
        pred_coefs = comm.allreduce(pred_coefs)  
        if gradient:
            grad_pred_coefs = comm.allreduce(grad_pred_coefs)  
 
    #print("pred time:", time.time()-predstart,flush=True)
    if inp.salted.verbose and rank==0:
        print(f"Total prediction time = {(time.time() - start_time):.2f} s", flush=True)
    
    if qmcode=="cp2k":

        lcuts = {}
        for spe in species:
            lcuts[spe] = min(lcut,lmax[spe])
 
        charge, dipole = compute_charge_and_dipole(structure,inp.qm.pseudocharge,natoms,atomic_symbols,lcuts,nmax,species,charge_integrals,dipole_integrals,pred_coefs,average)
        
        if gradient:

            grad_charge = scale_grad_coefs(structure,inp.qm.pseudocharge,natoms,atomic_symbols,lcuts,nmax,species,charge_integrals,pred_coefs,grad_pred_coefs,average,charge)

            return [pred_coefs, grad_pred_coefs, charge, dipole] 

        else:

            return [pred_coefs, charge, dipole] 

    else:

        if gradient:

            return [pred_coefs, grad_pred_coefs]

        else:

            return [pred_coefs]

if __name__ == "__main__":
    build()
