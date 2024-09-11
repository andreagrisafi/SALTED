"""
Calculate RKHS vectors
"""

import os
import os.path as osp
import time
from typing import Dict, List, Tuple
import copy

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from scipy import sparse

from salted import sph_utils
from salted.lib import equicomb, equicombsparse, antiequicomb, antiequicombsparse, equicombnonorm, antiequicombnonorm, kernelequicomb
from salted.sys_utils import ParseConfig, get_atom_idx, get_conf_range, get_feats_projs, get_feats_projs_response, read_system

def build():

    # salted parameters
    (saltedname, saltedpath, saltedtype,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

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

    # TODO: replace class arraylist with numpy.concatenate
    # define a numpy equivalent to an appendable list
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

    fdir = f"rkhs-vectors_{saltedname}"
    
    if (rank == 0):
        dirpath = os.path.join(saltedpath, fdir, f"M{Menv}_zeta{zeta}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    if size > 1:  comm.Barrier()

    # Distribute structures to tasks
    if parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = range(ndata)

    frames = read(filename,":")

    if saltedtype=="density":

        # Load feature space sparsification information if required
        if sparsify:
            vfps = {}
            for lam in range(lmax_max+1):
                vfps[lam] = np.load(osp.join(
                    saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
                ))
        
        # Load training feature vectors and RKHS projection matrix
        Vmat,Mspe,power_env_sparse = get_feats_projs(species,lmax)

        # compute the weight-vector size
        cuml_Mcut = {}
        totsize = 0
        for spe in species:
            for lam in range(lmax[spe]+1):
                for n in range(nmax[(spe,lam)]):
                    cuml_Mcut[(spe,lam,n)] = totsize
                    totsize += Vmat[(lam,spe)].shape[1]
    
        if rank == 0: print(f"problem dimensionality: {totsize}", flush=True)

        for iconf in conf_range:

            start_time = time.time()
            print(f"{iconf} start", flush=True)

            structure = frames[iconf]

            omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms[iconf])
            omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms[iconf])

            # Reshape arrays of expansion coefficients for optimal Fortran indexing
            v1 = np.transpose(omega1,(2,0,3,1))
            v2 = np.transpose(omega2,(2,0,3,1))

            # Compute equivariant features for the given structure
            power = {}
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
                    power[lam] = p.reshape(natoms[iconf],featsize)
                else:
                    power[lam] = p.reshape(natoms[iconf],2*lam+1,featsize)

            # Compute kernels and RKHS descriptors 
            Psi:Dict[Tuple[int, str], np.ndarray] = {}
            ispe = {}
            Tsize = 0
            for spe in species:

                ispe[spe] = 0

                # lam=0
                if zeta == 1:
                    # sparse power spectrum already projected on truncated RKHS
                    kernel0_nm = np.dot(power[0][atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                    Psi[(spe,0)] = kernel0_nm

                else:

                    kernel0_nm = np.dot(power[0][atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                    kernel_nm = kernel0_nm**zeta
                    Psi[(spe,0)] = np.real(np.dot(kernel_nm,Vmat[(0,spe)]))

                Tsize += natom_dict[(iconf,spe)]*nmax[(spe,0)]

                # lam>0
                for lam in range(1,lmax[spe]+1):

                    # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
                    if zeta == 1:

                        # sparse power spectrum already projected on truncated RKHS
                        Psi[(spe,lam)] = np.dot(power[lam][atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),power[lam].shape[-1]),power_env_sparse[(lam,spe)].T)

                    else:

                        kernel_nm = np.dot(power[lam][atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),power[lam].shape[-1]),power_env_sparse[(lam,spe)].T)
                        for i1 in range(natom_dict[(iconf,spe)]):
                            for i2 in range(Mspe[spe]):
                                kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[i1,i2]**(zeta-1)
                        Psi[(spe,lam)] = np.real(np.dot(kernel_nm,Vmat[(lam,spe)]))
                
                    Tsize += natom_dict[(iconf,spe)]*nmax[(spe,lam)]*(2*lam+1)

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
                    i2 = ispe[spe]*(2*l+1) + 2*l+1
                    x = Psi[(spe,l)][i1:i2]  # 2d array
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
            sparse.save_npz(osp.join(
                saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
            ), sparse_psi)

            del sparse_psi
            del psi_nonzero
            del ij

            end_time = time.time()
            print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)

    elif saltedtype=="density-response":

        # Load training feature vectors and RKHS projection matrix
        Vmat,Mspe,power_env_sparse,power_env_sparse_antisymm = get_feats_projs_response(species,lmax)

        # compute the weight-vector size
        cuml_Mcut = {}
        totsize = 0
        for spe in species:
            for lam in range(lmax[spe]+1):
                for n in range(nmax[(spe,lam)]):
                    cuml_Mcut[(spe,lam,n)] = totsize
                    totsize += Vmat[(lam,spe)].shape[1]
    
        if rank == 0: print(f"problem dimensionality: {totsize}", flush=True)

        lmax_max += 1

        cart = ["y","z","x"]

        for iconf in conf_range:

            start_time = time.time()
            print(f"{iconf} start", flush=True)

            structure = frames[iconf]

            omega1 = sph_utils.get_representation_coeffs(structure,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms[iconf])
            omega2 = sph_utils.get_representation_coeffs(structure,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms[iconf])

            # Reshape arrays of expansion coefficients for optimal Fortran indexing
            v1 = np.transpose(omega1,(2,0,3,1))
            v2 = np.transpose(omega2,(2,0,3,1))

            # Compute equivariant features for the given structure
            power = {}
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
                    power[lam] = p.reshape(natoms[iconf],featsize)
                else:
                    power[lam] = p.reshape(natoms[iconf],2*lam+1,featsize)

            # Compute antisymmetric equivariant features for the given structure
            power_antisymm = {}
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

                # Fill vector of equivariant descriptor
                power_antisymm[lam] = p.reshape(natoms[iconf],2*lam+1,featsize)

  
            # Compute kernels and RKHS descriptors
            Psi_cart = {}
            for ic in cart:
                for spe in species:
                    for lam in range(lmax[spe]+1):
                        Psi_cart[(ic,spe,lam)] = np.zeros((natom_dict[(iconf,spe)]*(2*lam+1),Vmat[(lam,spe)].shape[-1]))


            Psi:Dict[Tuple[int, str], np.ndarray] = {}
            ispe = {}
            Tsize = 0
            for spe in species:

                for icart in range(3):
                    ispe[(icart,spe)] = 0

                # lam=0
                kernel0_nm = np.dot(power[0][atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                kernel_nm = np.dot(power[1][atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*3,power[1].shape[-1]),power_env_sparse[(1,spe)].T)
                for i1 in range(natom_dict[(iconf,spe)]):
                    for i2 in range(Mspe[spe]):
                        kernel_nm[i1*3:i1*3+3][:,i2*3:i2*3+3] *= kernel0_nm[i1,i2]**(zeta-1)

                kernel0_nn = np.dot(power[0][atom_idx[(iconf,spe)]],power[0][atom_idx[(iconf,spe)]].T)
                kernel_nn = np.dot(power[1][atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*3,power[1].shape[-1]),power[1][atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*3,power[1].shape[-1]).T)
                normfact = np.zeros(natom_dict[(iconf,spe)])
                for i1 in range(natom_dict[(iconf,spe)]):
                    for i2 in range(natom_dict[(iconf,spe)]):
                        kernel_nn[i1*3:i1*3+3][:,i2*3:i2*3+3] *= kernel0_nn[i1,i2]**(zeta-1)
                    normfact[i1] = np.sqrt(np.sum(kernel_nn[i1*3:i1*3+3][:,i1*3:i1*3+3]**2))

                normfact_sparse = np.load(os.path.join(saltedpath, "normfacts", f"normfact_spe-{spe}_lam-{0}.npy"))
                j1 = 0
                for i1 in range(natom_dict[(iconf,spe)]):
                    norm1 = normfact[i1]
                    for imu1 in range(3):
                        j2 = 0
                        for i2 in range(Mspe[spe]):
                            norm2 = normfact_sparse[i2]
                            for imu2 in range(3):
                                kernel_nm[j1,j2] /= norm1*norm2
                                j2 += 1
                        j1 += 1

                Psi[(spe,0)] = np.real(np.dot(kernel_nm,Vmat[(0,spe)]))
               
                # normalize RKHS descriptor
                #psi = Psi[(spe,0)].reshape(natom_dict[(iconf,spe)],3,Vmat[(0,spe)].shape[-1]).reshape(natom_dict[(iconf,spe)],3*Vmat[(0,spe)].shape[-1])
                #inner = np.diagonal(np.dot(psi,psi.T))
                #Psi[(spe,0)] = np.einsum('a,ab->ab',1.0/np.sqrt(inner),psi).reshape(natom_dict[(iconf,spe)],3,Vmat[(0,spe)].shape[-1]).reshape(natom_dict[(iconf,spe)]*3,Vmat[(0,spe)].shape[-1])

                Tsize += natom_dict[(iconf,spe)]*nmax[(spe,0)]

                #TODO uncomment for covariance test
                #if spe=="O":
                #    if iconf==0:
                #        psi_1 = Psi[(spe,0)]
                #    if iconf==10:
                #        Dreal = np.load("Dreal_L1_from_10_to_0.npy")
                #        psi_2 = Psi[(spe,0)]
                #        psi_2_aligned = np.dot(Dreal,psi_2)
                #        print(psi_2_aligned-psi_1)
                #        sys.exit(0)

                idx = 0
                idx_cart = 0
                for iat in range(natom_dict[(iconf,spe)]):
                    for ik in range(3):
                        Psi_cart[(cart[ik],spe,0)][idx_cart] = Psi[(spe,0)][idx]
                        idx += 1
                    idx_cart += 1

                # lam>0
                for lam in range(1,lmax[spe]+1):

                    Msize = Mspe[spe]*3*(2*lam+1)
                    Nsize = natom_dict[(iconf,spe)]*3*(2*lam+1)
                    kernel_nm = np.zeros((Nsize,Msize),complex)
                    kernel_nn = np.zeros((Nsize,Nsize),complex)

                    # Perform CG combination
                    for L in [lam-1,lam,lam+1]:

                        #print("L=", L)

                        c2r = sph_utils.complex_to_real_transformation([2*L+1])[0]

                        # compute complex descriptor for the given L
                        if L==lam:
                            pimag = power_antisymm[L][atom_idx[(iconf,spe)]]
                            featsize = pimag.shape[-1]
                            pimag = pimag.reshape(natom_dict[(iconf,spe)],2*L+1,featsize)
                            pimag = np.transpose(pimag,(1,0,2)).reshape(2*L+1,natom_dict[(iconf,spe)]*featsize)
                            preal = np.zeros_like(pimag)
                        else:
                            preal = power[L][atom_idx[(iconf,spe)]]
                            featsize = preal.shape[-1]
                            preal = preal.reshape(natom_dict[(iconf,spe)],2*L+1,featsize)
                            preal = np.transpose(preal,(1,0,2)).reshape(2*L+1,natom_dict[(iconf,spe)]*featsize)
                            pimag = np.zeros_like(preal)

                        ptemp = preal + 1j * pimag
                        pcmplx = np.dot(np.conj(c2r.T),ptemp).reshape(2*L+1,natom_dict[(iconf,spe)],featsize)
                        pcmplx = np.transpose(pcmplx,(1,0,2)).reshape(natom_dict[(iconf,spe)]*(2*L+1),featsize)

                        # compute complex sparse descriptor for the given L 
                        if L==lam:
                            pimag = power_env_sparse_antisymm[(L,spe)]
                            featsize = pimag.shape[-1]
                            pimag = pimag.reshape(Mspe[spe],2*L+1,featsize)
                            pimag = np.transpose(pimag,(1,0,2)).reshape(2*L+1,Mspe[spe]*featsize)
                            preal = np.zeros_like(pimag)
                        else:
                            preal = power_env_sparse[(L,spe)]
                            featsize = preal.shape[-1]
                            preal = preal.reshape(Mspe[spe],2*L+1,featsize)
                            preal = np.transpose(preal,(1,0,2)).reshape(2*L+1,Mspe[spe]*featsize)
                            pimag = np.zeros_like(preal)

                        ptemp = preal + 1j * pimag
                        pcmplx_sparse = np.dot(np.conj(c2r.T),ptemp).reshape(2*L+1,Mspe[spe],featsize)
                        pcmplx_sparse = np.transpose(pcmplx_sparse,(1,0,2)).reshape(Mspe[spe]*(2*L+1),featsize)

                        # compute complex K_nm kernel 
                        knm = np.dot(pcmplx,np.conj(pcmplx_sparse).T)

                        # load the relevant CG coefficients 
                        cgcoefs = np.loadtxt(os.path.join(saltedpath, "wigners", f"cg_response_lam-{lam}_L-{L}.dat"))

                        k0 = kernel0_nm**(zeta-1)
                        cgkernel = kernelequicomb.kernelequicomb(natom_dict[(iconf,spe)],Mspe[spe],lam,1,L,Nsize,Msize,len(cgcoefs),cgcoefs,knm.T,k0.T)
                        kernel_nm += cgkernel.T
                        
                        # compute complex K_nn kernel 
                        knn = np.dot(pcmplx,np.conj(pcmplx).T)
                        k0 = kernel0_nn**(zeta-1)
                        cgkernel = kernelequicomb.kernelequicomb(natom_dict[(iconf,spe)],natom_dict[(iconf,spe)],lam,1,L,Nsize,Nsize,len(cgcoefs),cgcoefs,knn.T,k0.T)
                        kernel_nn += cgkernel.T
                        

                    # compute complex to real transformation matrix for lam X 1 tensor product space
                    A = sph_utils.complex_to_real_transformation([2*lam+1])[0]
                    B = sph_utils.complex_to_real_transformation([3])[0]
                    c2r = np.zeros((3*(2*lam+1),3*(2*lam+1)),complex)
                    j1 = 0
                    for i1 in range(2*lam+1):
                        j2 = 0
                        for i2 in range(2*lam+1):
                            c2r[j1:j1+3,j2:j2+3] = A[i1,i2] * B
                            j2 += 3
                        j1 += 3

                    # make kernel real
                    ktemp1 = np.dot(c2r,np.transpose(kernel_nm.reshape(natom_dict[(iconf,spe)],3*(2*lam+1),Msize),(1,0,2)).reshape(3*(2*lam+1),natom_dict[(iconf,spe)]*Msize))
                    ktemp2 = np.transpose(ktemp1.reshape(3*(2*lam+1),natom_dict[(iconf,spe)],Msize),(1,0,2)).reshape(Nsize,Msize)
                    kernel_nm = np.dot(ktemp2.reshape(Nsize,Mspe[spe],3*(2*lam+1)).reshape(Nsize*Mspe[spe],3*(2*lam+1)),np.conj(c2r).T).reshape(Nsize,Mspe[spe],3*(2*lam+1)).reshape(Nsize,Msize)
                    #print("imag:", np.linalg.norm(np.imag(kernel_nm)))

                    ktemp1 = np.dot(c2r,np.transpose(kernel_nn.reshape(natom_dict[(iconf,spe)],3*(2*lam+1),Nsize),(1,0,2)).reshape(3*(2*lam+1),natom_dict[(iconf,spe)]*Nsize))
                    ktemp2 = np.transpose(ktemp1.reshape(3*(2*lam+1),natom_dict[(iconf,spe)],Nsize),(1,0,2)).reshape(Nsize,Nsize)
                    kernel_nn = np.dot(ktemp2.reshape(Nsize,natom_dict[(iconf,spe)],3*(2*lam+1)).reshape(Nsize*natom_dict[(iconf,spe)],3*(2*lam+1)),np.conj(c2r).T).reshape(Nsize,natom_dict[(iconf,spe)],3*(2*lam+1)).reshape(Nsize,Nsize)

                    normfact = np.zeros(natom_dict[(iconf,spe)])
                    for i1 in range(natom_dict[(iconf,spe)]):
                        normfact[i1] = np.sqrt(np.sum(np.real(kernel_nn)[i1*3*(2*lam+1):i1*3*(2*lam+1)+3*(2*lam+1)][:,i1*3*(2*lam+1):i1*3*(2*lam+1)+3*(2*lam+1)]**2))

                    normfact_sparse = np.load(os.path.join(saltedpath, "normfacts", f"normfact_spe-{spe}_lam-{lam}.npy"))
                    j1 = 0 
                    for i1 in range(natom_dict[(iconf,spe)]):
                        norm1 = normfact[i1]
                        for imu1 in range(3*(2*lam+1)):
                            j2 = 0 
                            for i2 in range(Mspe[spe]):
                                norm2 = normfact_sparse[i2]
                                for imu2 in range(3*(2*lam+1)):
                                    kernel_nm[j1,j2] /= norm1*norm2
                                    j2 += 1
                            j1 += 1

                    # project kernel on the RKHS
                    Psi[(spe,lam)] = np.real(np.dot(np.real(kernel_nm),Vmat[(lam,spe)]))

                    # normalize RKHS descriptor
                    #psi = Psi[(spe,lam)].reshape(natom_dict[(iconf,spe)],3*(2*lam+1),Vmat[(lam,spe)].shape[-1]).reshape(natom_dict[(iconf,spe)],3*(2*lam+1)*Vmat[(lam,spe)].shape[-1])
                    #inner = np.diagonal(np.dot(psi,psi.T))
                    #Psi[(spe,lam)] = np.einsum('a,ab->ab',1.0/np.sqrt(inner),psi).reshape(natom_dict[(iconf,spe)],3*(2*lam+1),Vmat[(lam,spe)].shape[-1]).reshape(natom_dict[(iconf,spe)]*3*(2*lam+1),Vmat[(lam,spe)].shape[-1])

                    Tsize += natom_dict[(iconf,spe)]*(2*lam+1)*nmax[(spe,lam)]

                    #TODO uncomment for covariance test
                    #if spe=="H" and lam==1:
                    #    if iconf==0:
                    #        #psi_1 = Psi[(spe,lam)].reshape(9,Psi[(spe,lam)].shape[-1])
                    #        psi_1 = np.sum(Psi[(spe,lam)].reshape(2,9,Psi[(spe,lam)].shape[-1]),axis=0)
                    #    if iconf==10:
                    #        Dreal = np.load("Dreal_L1_from_10_to_0.npy")
                    #        D2real = np.zeros((9,9))
                    #        j1 = 0
                    #        for i1 in range(2*lam+1):
                    #            j2 = 0
                    #            for i2 in range(2*lam+1):
                    #                D2real[j1:j1+3,j2:j2+3] = Dreal[i1,i2] * Dreal
                    #                j2 += 3
                    #            j1 += 3
                    #        #psi_2 = Psi[(spe,lam)].reshape(9,Psi[(spe,lam)].shape[-1])
                    #        psi_2 = np.sum(Psi[(spe,lam)].reshape(2,9,Psi[(spe,lam)].shape[-1]),axis=0)
                    #        psi_2_aligned = np.dot(D2real,psi_2)
                    #        print(psi_1)
                    #        print(psi_2_aligned-psi_1)
                    #        sys.exit(0)

                    idx = 0
                    idx_cart = 0
                    for iat in range(natom_dict[(iconf,spe)]):
                        for imu in range(2*lam+1):
                            for ik in range(3):
                                Psi_cart[(cart[ik],spe,lam)][idx_cart] = Psi[(spe,lam)][idx]
                                idx += 1
                            idx_cart += 1

                    #for ik in range(3):
                    #    print(spe,lam,cart[ik],np.linalg.norm(Psi_cart[(cart[ik],spe,lam)]))


            # build sparse feature-vector memory efficiently
            nrows = Tsize
            ncols = totsize

            for icart in range(3):

                srows = arraylist()
                scols = arraylist()
                psi_nonzero = arraylist()

                i = 0
                for iat in range(natoms[iconf]):
                    spe = atomic_symbols[iconf][iat]
                    for l in range(lmax[spe]+1):
                        i1 = ispe[(icart,spe)]*(2*l+1)
                        i2 = ispe[(icart,spe)]*(2*l+1) + 2*l+1
                        x = Psi_cart[(cart[icart],spe,l)][i1:i2]  # 2d array
                        nz = np.nonzero(x)  # rwo 0: non-zero row indices, row 1: non-zero column indices
                        vals = x[nz]  # 1d array
                        for n in range(nmax[(spe,l)]):
                            psi_nonzero.update(vals)
                            srows.update(nz[0]+i)
                            scols.update(nz[1]+cuml_Mcut[(spe,l,n)])
                            i += (2*l+1)
                    ispe[(icart,spe)] += 1

                psi_nonzero = psi_nonzero.finalize()
                srows = srows.finalize()
                scols = scols.finalize()
                ij = np.vstack((srows,scols))
    
                del srows
                del scols
    
                sparse_psi = sparse.coo_matrix((psi_nonzero, ij), shape=(nrows, ncols))
                sparse.save_npz(osp.join(
                    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{cart[icart]}.npz"
                ), sparse_psi)
    
                del sparse_psi
                del psi_nonzero
                del ij

            end_time = time.time()
            print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)

if __name__ == "__main__":
    build()
