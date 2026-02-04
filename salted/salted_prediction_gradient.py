import os
import sys
import time
import h5py
import numpy as np
from scipy import special
from ase.data import atomic_numbers
from ase.io import read

from salted.lib import equicomb_grad
from salted.lib import equicombsparse_grad

from salted import sph_utils
from salted import basis
from salted.sys_utils import ParseConfig

def build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,structure):
#def build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,structure,rank,ntasks):
    
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

    pseudocharge = inp.qm.pseudocharge
    pseudocharge_dict = {}
    for i in range(len(species)):
        pseudocharge_dict[species[i]] = pseudocharge[i] # Warning: species and pseudocharge must have the same ordering

    omega1, domega1 = sph_utils.get_representation_gradient_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe1,species,nang1,nrad1,natoms)
    omega2, domega2 = sph_utils.get_representation_gradient_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe2,species,nang2,nrad2,natoms)

    #np.save(os.path.join(saltedpath, "omegas", f"omega1"), omega1)
    #np.save(os.path.join(saltedpath, "omegas", f"domega1"), domega1)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing 
    print(omega1.shape, domega1.shape)
    v1 = np.transpose(omega1,(2,0,3,1))
    dv1 = np.transpose(domega1.reshape((domega1.shape[0],natoms,natoms,3,domega1.shape[3],domega1.shape[4])),(2,3,4,0,5,1))
    #dv1 = np.transpose(domega1.reshape((domega1.shape[0],natoms,natoms,3,domega1.shape[3],domega1.shape[4])),(3,4,0,5,2,1))
    v2 = np.transpose(omega2,(2,0,3,1))
    dv2 = np.transpose(domega2.reshape((domega2.shape[0],natoms,natoms,3,domega2.shape[3],domega2.shape[4])),(2,3,4,0,5,1))
    #dv2 = np.transpose(domega2.reshape((domega2.shape[0],natoms,natoms,3,domega2.shape[3],domega2.shape[4])),(3,4,0,5,2,1))
    v1 = np.asfortranarray(v1, dtype=np.complex128)
    dv1 = np.asfortranarray(dv1, dtype=np.complex128)
    v2 = np.asfortranarray(v2, dtype=np.complex128)
    dv2 = np.asfortranarray(dv2, dtype=np.complex128)

    #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/v1", v1)
    #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/dv1", dv1)

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    pvec = {}
    grad_pvec = {}
    #for lam in range(1):
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
            start = time.time()
            print("before")
            p, grad_p = equicombsparse_grad.equicombsparse_grad(natoms,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,dv1,dv2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize,nfps,vfps[lam])

            end = time.time()
            print(str(end-start))
            
            print("done1")

            p = np.transpose(p,(2,0,1))
            grad_p = np.transpose(grad_p,(0,1,4,2,3))
            #grad_p = np.transpose(grad_p,(4,3,0,1,2))
            featsize = ncut

            #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/p", p)
            #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/grad_p", grad_p)

        else:

            featsize = nspe1*nspe2*nrad1*nrad2*llmax
            print("hi1")
            p, grad_p = equicomb_grad.equicomb_grad(natoms,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,dv1,dv2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
            print("done1")
            p = np.transpose(p,(2,0,1))
            grad_p = np.transpose(grad_p,(0,1,4,2,3))

        if lam==0: 
            pvec[lam] = p.reshape(natoms,featsize)
            grad_pvec[lam] = grad_p.reshape(natoms,3,natoms,featsize)
        else:
            pvec[lam] = p.reshape(natoms,2*lam+1,featsize)
            grad_pvec[lam] = grad_p.reshape(natoms,3,natoms,2*lam+1,featsize)
        
        # print("equicomb time:", (time.time()-equistart))
    
    rkhsstart = time.time()
 
    psi_nm = {}
    grad_psi_nm = {}
    for spe in species:

        # lam = 0
        if zeta==1:
            psi_nm[(spe,0)] = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
            grad_psi_nm[(spe,0)] = np.dot(grad_pvec[0][:,:,atom_idx[spe],:].reshape(natoms*3*len(atom_idx[spe]),featsize),power_env_sparse[(0,spe)].T)
            #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/p", psi_nm[(spe,0)])
            #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/grad_p", grad_psi_nm[(spe,0)].reshape((len(atom_idx[spe]),natoms,3,power_env_sparse[(0,spe)].T.shape[1])))
        else:
            kernel0_nm = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
            grad_kernel0_nm = np.dot(grad_pvec[0][:,:,atom_idx[spe],:].reshape((natoms*3*len(atom_idx[spe]),featsize)),power_env_sparse[(0,spe)].T)
            kernel_nm = kernel0_nm**zeta
            grad_kernel_nm = zeta*grad_kernel0_nm**(zeta-1)
            psi_nm[(spe,0)] = np.dot(kernel_nm,Vmat[(0,spe)])
            grad_psi_nm[(spe,0)] = np.dot(grad_kernel_nm,Vmat[(0,spe)])

        # lam > 0
        #for lam in range(1,1):
        for lam in range(1,lmax[spe]+1):

            featsize = pvec[lam].shape[-1]
            if zeta==1:
                psi_nm[(spe,lam)] = np.dot(pvec[lam][atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                grad_psi_nm[(spe,lam)] = np.dot(grad_pvec[lam][:,:,atom_idx[spe],:,:].reshape(natoms*3*natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                print("done2")
            else:
                kernel_nm = np.dot(pvec[lam][atom_idx[spe]].reshape(natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                kernel_nm_blocks = kernel_nm.reshape(natom_dict[spe], 2*lam+1, Mspe[spe], 2*lam+1)
                kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (zeta - 1)

                grad_kernel_nm = np.dot(grad_pvec[lam][:,:,atom_idx[spe],:,:].reshape(natoms*3*natom_dict[spe]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                grad_kernel_nm_blocks = grad_kernel_nm.reshape(natoms, 3, natom_dict[spe], 2*lam+1, Mspe[spe], 2*lam+1)
                grad_kernel_nm_blocks = (grad_kernel_nm_blocks*kernel0_nm[np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis] ** (zeta - 1)) + kernel_nm[np.newaxis, np.newaxis, :, :, :, :]*(zeta-1)*grad_kernel0_nm.reshape(natoms,3,(len(atom_idx[spe]),featsize))[:, :, :, np.newaxis, :, np.newaxis]**(zeta-2)
                
                kernel_nm = kernel_nm_blocks.reshape(natom_dict[spe]*(2*lam+1), Mspe[spe]*(2*lam+1))
                grad_kernel_nm = grad_kernel_nm_blocks.reshape(natoms*3*natom_dict[spe]*(2*lam+1), Mspe[spe]*(2*lam+1))
                #for i1 in range(natom_dict[spe]):
                #    for i2 in range(Mspe[spe]):
                #        kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[i1,i2]**(zeta-1)
                psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])
                grad_psi_nm[(spe,lam)] = np.dot(grad_kernel_nm,Vmat[(lam,spe)])

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
        #for l in range(1):
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Tsize += 2*l+1
    
    # compute predictions per channel
    C = {}
    grad_C = {}
    ispe = {}
    isize = 0
    for spe in species:
        ispe[spe] = 0
        #for l in range(1):
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = psi_nm[(spe,l)].shape[1]
                print(Mcut)
                C[(spe,l,n)] = np.dot(psi_nm[(spe,l)],weights[isize:isize+Mcut])
                grad_C[(spe,l,n)] = np.dot(grad_psi_nm[(spe,l)],weights[isize:isize+Mcut])
                isize += Mcut

    l = 2
    ave_c_n = np.zeros(nmax[(species[0],l)])
    ave_grad_c_n = np.zeros((nmax[(species[0],l)], natoms, 3))
    for n in range(nmax[(species[0],l)]):
        ave_c_n[n] = np.mean(C[(species[0],l,n)])
        ave_grad_c_n[n] = np.mean(grad_C[(species[0],l,n)].reshape((natoms,3,len(atom_idx[species[0]])*(2*l+1))), axis = 2)

    ave_c_n = np.array([C[(species[0],l,n)] for n in range(nmax[(species[0],l)])])
    ave_grad_c_n = np.array([grad_C[(species[0],l,n)].reshape((natoms,3,len(atom_idx[species[0]])*(2*l+1))) for n in range(nmax[(species[0],l)])])
    np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/ave_c_n_2", ave_c_n)
    np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/ave_grad_c_n_2", ave_grad_c_n)

    print(C[(species[0],2,0)].shape)
    print(grad_C[(species[0],2,0)].shape)

    # init averages array if asked
    if average:
        Av_coeffs = np.zeros(Tsize)
    
    # fill vector of predictions
    i = 0
    i_grad = 0
    pred_coefs = np.zeros(Tsize)
    grad_pred_coefs = np.zeros(Tsize*natoms*3)
    dict_grad_pred_coefs = {}
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        #for l in range(1):
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                grad_pred_coefs[i_grad:i_grad+(2*l+1)*natoms*3] = grad_C[(spe,l,n)][ispe[spe]*(2*l+1)*natoms*3:ispe[spe]*(2*l+1)*natoms*3+(2*l+1)*natoms*3]
                i += 2*l+1
                i_grad += (2*l+1)*natoms*3
                if average and l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
        ispe[spe] += 1

    print("done3")
    
    #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/pred_coeffs", pred_coefs)
    #np.save("/lus/home/CT9/c1710463/tbernhard/qmmm_salted/pt100_water/dataset_pt100/grad_pred_coeffs", grad_pred_coefs)

    # add back spherical averages if required
    if average:
        pred_coefs += Av_coeffs
    
    if qmcode=="cp2k":

        # compute integral of predicted density
        iaux = 0
        rho_int = 0.0
        nele = 0.0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            if average:
                nele += pseudocharge_dict[spe]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                        rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                    iaux += 2*l+1
 
        print("charge integral =", rho_int) 


        # enforce charge conservation 
        iaux = 0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0 and im==0:
                            if average:
                                pred_coefs[iaux] *= nele/rho_int
                            else:
                                if n==nmax[(spe,l)]-1:
                                    pred_coefs[iaux] -= rho_int/(charge_integrals[(spe,l,n)]*natoms)
                        iaux += 1

 
#    if print("pred time:", time.time()-predstart,flush=True)
    
    return pred_coefs, ave_c_n, ave_grad_c_n

if __name__ == "__main__":
    build()
