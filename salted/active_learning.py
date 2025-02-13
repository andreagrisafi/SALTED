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
from salted.sys_utils import ParseConfig

def build(lmax,nmax,lmax_max,power_env_sparse,Mspe,Vmat,vfps,ntrain,Avec,Bmat,ref_coefs,over,structure):

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
   
    # compute the weight-vector size
    cuml_Mcut = {}
    totsize = 0
    for spe in species:
        for lam in range(lmax[spe]+1):
            for n in range(nmax[(spe,lam)]):
                cuml_Mcut[(spe,lam,n)] = totsize
                totsize += Vmat[(lam,spe)].shape[1]
 
    omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe1,species,nang1,nrad1,natoms)
    omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe2,species,nang2,nrad2,natoms)

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
            p = equicombsparse.equicombsparse(natoms,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize,nfps,vfps[lam])
            p = np.transpose(p,(2,0,1))
            featsize = ncut

        else:

            featsize = nspe1*nspe2*nrad1*nrad2*llmax
            p = equicomb.equicomb(natoms,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
            p = np.transpose(p,(2,0,1))

        if lam==0: 
            pvec[lam] = p.reshape(natoms,featsize)
        else:
            pvec[lam] = p.reshape(natoms,2*lam+1,featsize)
        
        # print("equicomb time:", (time.time()-equistart))
    
    rkhsstart = time.time()
 
    Tsize = 0
    psi_nm = {}
    for spe in species:

        # lam = 0
        if zeta==1:
            psi_nm[(spe,0)] = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
        else:
            kernel0_nm = np.dot(pvec[0][atom_idx[spe]],power_env_sparse[(0,spe)].T)
            kernel_nm = kernel0_nm**zeta
            psi_nm[(spe,0)] = np.dot(kernel_nm,Vmat[(0,spe)])

        Tsize += natom_dict[spe]*nmax[(spe,0)]

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
                psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])
            
            Tsize += natom_dict[spe]*nmax[(spe,lam)]*(2*lam+1)

    # build sparse feature-vector memory efficiently
    nrows = Tsize
    ncols = totsize
    srows = arraylist()
    scols = arraylist()
    psi_nonzero = arraylist()

    i = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(lmax[spe]+1):
            i1 = ispe[spe]*(2*l+1)
            i2 = ispe[spe]*(2*l+1) + 2*l+1
            x = psi_nm[(spe,l)][i1:i2]  # 2d array
            nz = np.nonzero(x)  # rwo 0: non-zero row indices, row 1: non-zero column indices
            vals = x[nz]  # 1d array
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

    del srows
    del scols

    sparse_psi = sparse.coo_matrix((psi_nonzero, ij), shape=(nrows, ncols))
    #sparse.save_npz(osp.join(
    #    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
    #), sparse_psi)

    del psi_nonzero
    del ij

    psi = sparse_psi.toarray()
    del sparse_psi

    if average:

        # fill array of average spherical components
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            if spe in species:
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if l==0:
                           Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1

        # subtract average
        ref_coefs -= Av_coeffs

    ref_projs = np.dot(over,ref_coefs)

    # rescale property vector and Hessian matrix by the old number of training points
    Avec *= ntrain
    Bmat *= ntrain
    
    # updated property vector and Hessian matrix with the new training point 
    Avec += np.dot(psi.T,ref_projs)
    Bmat += np.dot(psi.T,np.dot(over,psi))
    
    # normalize property vector and Hessian matrix by the total new number of training points
    Avec /= float(ntrain+1)
    Bmat /= float(ntrain+1)

    #np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{ntrain+1}.npy"), Avec)
    #np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{ntrain+1}.npy"), Bmat)

    # compute new regression weights
    w = np.linalg.solve(Bmat+np.eye(totsize)*regul,Avec)

    #np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(regul))}.npy"), w)

    return w

if __name__ == "__main__":
    build()
