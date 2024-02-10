"""
WARNING: many variables are referenced before defined
"""

import os
import sys
import time
import h5py
import numpy as np
from scipy import special
from ase.data import atomic_numbers
from ase.io import read

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from metatensor import Labels

from salted.lib import equicomb 
from salted.lib import equicombfield 

from salted import sph_utils
from salted import basis
from salted import efield

def build(lmax,nmax,lmax_max,weights,power_env_sparse,Vmat,vfps,charge_integrals,structure):

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
    rep1 = inp.rep1 
    rcut1 = inp.rcut1
    sig1 = inp.sig1
    nrad1 = inp.nrad1
    nang1 = inp.nang1
    neighspe1 = inp.neighspe1
    rep2 = inp.rep2
    rcut2 = inp.rcut2
    sig2 = inp.sig2
    nrad2 = inp.nrad2
    nang2 = inp.nang2
    neighspe2 = inp.neighspe2
    ncut = inp.ncut
    species = inp.species
    M = inp.Menv
    zeta = inp.z
    reg = inp.regul
    
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
   
    atom_idx = {}
    natom_dict = {}
    for spe in species:
        atom_idx[spe] = []
        natom_dict[spe] = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if spe in species:
           atom_idx[spe].append(iat)
           natom_dict[spe] += 1
    
    HYPER_PARAMETERS_DENSITY = {
        "cutoff": rcut1,
        "max_radial": nrad1,
        "max_angular": nang1,
        "atomic_gaussian_width": sig1,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    }
    
    HYPER_PARAMETERS_POTENTIAL = {
        "potential_exponent": 1,
        "cutoff": rcut2,
        "max_radial": nrad2,
        "max_angular": nang2,
        "atomic_gaussian_width": sig2,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}}
    }
    
    if rep1=="rho":
        # get SPH expansion for atomic density    
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)
    
    elif rep1=="V":
        # get SPH expansion for atomic potential 
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)
    
    else:
        if rank == 0: print("Error: requested representation", rep1, "not provided")
    
    descstart = time.time()
    
    nspe1 = len(neighspe1)
    keys_array = np.zeros(((nang1+1)*len(species)*nspe1,3),int)
    i = 0
    for l in range(nang1+1):
        for specen in species:
            for speneigh in neighspe1:
                keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1
    
    keys_selection = Labels(
        names=["spherical_harmonics_l","species_center","species_neighbor"],
        values=keys_array
    )
    
    spx = calculator.compute(structure, selected_keys=keys_selection)
    spx = spx.keys_to_properties("species_neighbor")
    spx = spx.keys_to_samples("species_center")
    
    # Get 1st set of coefficients as a complex numpy array
    omega1 = np.zeros((nang1+1,natoms,2*nang1+1,nspe1*nrad1),complex)
    for l in range(nang1+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(spherical_harmonics_l=l).values)
    
    if rep2=="rho":
        # get SPH expansion for atomic density    
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)
    
    elif rep2=="V":
        # get SPH expansion for atomic potential 
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL) 
    
    else:
        if rank == 0: print("Error: requested representation", rep2, "not provided")
    
    nspe2 = len(neighspe2)
    keys_array = np.zeros(((nang2+1)*len(species)*nspe2,3),int)
    i = 0
    for l in range(nang2+1):
        for specen in species:
            for speneigh in neighspe2:
                keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1
    
    keys_selection = Labels(
        names=["spherical_harmonics_l","species_center","species_neighbor"],
        values=keys_array
    )
    
    spx_pot = calculator.compute(structure, selected_keys=keys_selection)
    spx_pot = spx_pot.keys_to_properties("species_neighbor")
    spx_pot = spx_pot.keys_to_samples("species_center")
    
    # Get 2nd set of coefficients as a complex numpy array 
    omega2 = np.zeros((nang2+1,natoms,2*nang2+1,nspe2*nrad2),complex)
    for l in range(nang2+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(spherical_harmonics_l=l).values)
    
    if inp.field:
    # get SPH expansion for a uniform and constant external field aligned along Z 
        omega_field = np.zeros((natoms,nrad2),complex)
        for iat in range(natoms):
            omega_field[iat] = efield.get_efield_sph(nrad2,rcut2)
    
#    if rank == 0: print("coefficients time:", (time.time()-descstart))
#    if rank == 0: print("")
    
    if size > 1: comm.Barrier()
    
    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    psi_nm = {}
    for lam in range(lmax_max+1):
    
#        if rank == 0: print("lambda =", lam)
    
        equistart = time.time()
    
        # Select relevant angular components for equivariant descriptor calculation
        llmax = 0
        lvalues = {}
        for l1 in range(nang1+1):
            for l2 in range(nang2+1):
                # keep only even combination to enforce inversion symmetry
                if (lam+l1+l2)%2==0 :
                    if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                        lvalues[llmax] = [l1,l2]
                        llmax+=1
        # Fill dense array from dictionary
        llvec = np.zeros((llmax,2),int)
        for il in range(llmax): 
            llvec[il,0] = lvalues[il][0]
            llvec[il,1] = lvalues[il][1]
        
        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(os.path.join(
            inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
        ))
        wigdim = wigner3j.size
      
        # Reshape arrays of expansion coefficients for optimal Fortran indexing 
        v1 = np.transpose(omega1,(2,0,3,1))
        v2 = np.transpose(omega2,(2,0,3,1))
    
        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]
    
        # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
        p = equicomb.equicomb(natoms,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
    
        # Define feature space and reshape equivariant descriptor
        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        p = np.transpose(p,(4,0,1,2,3)).reshape(natoms,2*lam+1,featsize)
     
#        if rank == 0: print("equivariant time:", (time.time()-equistart))
        
        normstart = time.time()
        
        # Normalize equivariant descriptor  
        inner = np.einsum('ab,ab->a', p.reshape(natoms,(2*lam+1)*featsize),p.reshape(natoms,(2*lam+1)*featsize))
        p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
        
#        if rank == 0: print("norm time:", (time.time()-normstart))
    
        sparsestart = time.time()
        
        if ncut > -1:
            p = p.reshape(natoms*(2*lam+1),featsize)
            p = p.T[vfps[lam]].T
            featsize = inp.ncut
        
#        if rank == 0: print("sparse time:", (time.time()-sparsestart))
        
        fillstart = time.time()
    
        # Fill vector of equivariant descriptor 
        if lam==0:
            pvec = p.reshape(natoms,featsize)
        else:
            pvec = p.reshape(natoms,2*lam+1,featsize)
    
#        if rank == 0: print("fill vector time:", (time.time()-fillstart))
    
        if inp.field:
             #########################################################
             #                 START E-FIELD HERE
             #########################################################
          
             # Select relevant angular components for equivariant descriptor calculation
             llmax = 0
             lvalues = {}
             for l1 in range(nang1+1):
                 # keep only even combination to enforce inversion symmetry
                 if (lam+l1+1)%2==0 :
                     if abs(1-lam) <= l1 and l1 <= (1+lam) :
                         lvalues[llmax] = [l1,1]
                         llmax+=1
             # Fill dense array from dictionary
             llvec = np.zeros((llmax,2),int)
             for il in range(llmax):
                 llvec[il,0] = lvalues[il][0]
                 llvec[il,1] = lvalues[il][1]
    
             # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
             wigner3j = np.loadtxt(os.path.join(
                 inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_field.dat"
             ))
             wigdim = wigner3j.size
          
             # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
             v2 = omega_field.T
             p = equicombfield.equicombfield(natoms,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
     
             # Define feature space and reshape equivariant descriptor
             featsizefield = nspe1*nrad1*nrad2*llmax
             p = np.transpose(p,(4,0,1,2,3)).reshape(natoms,2*lam+1,featsizefield)
           
#             if rank == 0: print("field equivariant time:", (time.time()-equistart))
              
             normstart = time.time()
     
             # Normalize equivariant descriptor  
             inner = np.einsum('ab,ab->a', p.reshape(natoms,(2*lam+1)*featsizefield),p.reshape(natoms,(2*lam+1)*featsizefield))
             p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
        
#             if rank == 0: print("field norm time:", (time.time()-normstart))
    
             if ncut > -1:
                 p = p.reshape(natoms*(2*lam+1),featsizefield)
                 p = p.T[vfps_field[lam]].T
                 featsizefield = inp.ncut 
    
             fillstart = time.time()
     
             # Fill vector of equivariant descriptor 
             if lam==0:
                 pvec = p.reshape(natoms,featsizefield)
             else:
                 pvec = p.reshape(natoms,2*lam+1,featsizefield)
    
        rkhsstart = time.time()
 
        if lam==0:
    
            if zeta==1: 
                 # Compute scalar kernels
                 kernel0_nm = {}
                 for spe in species:
                     if inp.field: 
                         psi_nm[(spe,lam)] = np.dot(pvec_field[atom_idx[spe]],power_env_sparse_field[(lam,spe)].T) 
                     else:
                         psi_nm[(spe,lam)] = np.dot(pvec[atom_idx[spe]],power_env_sparse[(lam,spe)].T)
            else:
                 # Compute scalar kernels
                 kernel0_nm = {}
                 for spe in species:
                     kernel0_nm[spe] = np.dot(pvec[atom_idx[spe]],power_env_sparse[(lam,spe)].T)
                     if inp.field:
                         kernel_nm = np.dot(pvec_field[atom_idx[spe]],power_env_sparse_field[(lam,spe)].T) * kernel0_nm[spe]**(zeta-1)
                     else:
                         kernel_nm = kernel0_nm[spe]**zeta
                     # Project on RKHS
                     psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])    
 
        else:
  
            if zeta==1: 
                # Compute covariant kernels
                for spe in species:
                    if inp.field: 
                        psi_nm[(spe,lam)] = np.dot(pvec_field[atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),pvec_field.shape[-1]),power_env_sparse_field[(lam,spe)].T)
                    else:
                        psi_nm[(spe,lam)] = np.dot(pvec[atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
            else: 
                # Compute covariant kernels
                for spe in species:
                    if inp.field:
                        kernel_nm = np.dot(pvec_field[atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),pvec_field.shape[-1]),power_env_sparse_field[(lam,spe)].T)
                    else:
                        kernel_nm = np.dot(pvec[atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                    for i1 in range(natom_dict[spe]):
                        for i2 in range(Mspe[spe]):
                            kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[spe][i1,i2]**(zeta-1)
                    # Project on RKHS
                    psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])
    
#        if rank == 0: print("rkhs time:", time.time()-rkhsstart,flush=True)
    
    # Perform equivariant predictions
    predstart = time.time()
    
    # Load spherical averages if required
    if inp.average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load("averages_"+str(spe)+".npy")
    
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
    if inp.average:
        Av_coeffs = np.zeros(Tsize)
    
    # fill vector of predictions
    i = 0
    pred_coefs = np.zeros(Tsize)
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                if inp.average and l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1
    
    # add back spherical averages if required
    if inp.average:
        pred_coefs += Av_coeffs
    
   
    if inp.qmcode=="cp2k":

        # compute integral of predicted density
        iaux = 0
        rho_int = 0.0
        nele = 0.0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            if inp.average:
                nele += inp.pseudocharge
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                        rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                    iaux += 2*l+1


        # enforce charge conservation 
        iaux = 0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0 and im==0:
                            if inp.average:
                                pred_coefs[iaux] *= nele/rho_int
                            else:
                                if n==nmax[(spe,l)]-1:
                                    pred_coefs[iaux] -= rho_int/(charge_integrals[(spe,l,n)]*natoms)
                        iaux += 1

 
#    if rank == 0: print("pred time:", time.time()-predstart,flush=True)
    
    return pred_coefs

if __name__ == "__main__":
    build()
