"""
Calculate RKHS projection matrix
"""

import os
import sys

import h5py
import numpy as np
from sympy.physics.wigner import wigner_3j

from salted.lib import kernelequicomb, kernelnorm
from salted import sph_utils
from salted.sys_utils import ParseConfig, get_atom_idx, read_system, rkhs_proj


def build():
    # inp = ParseConfig().parse_input()  # not used for now

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

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    sdir = os.path.join(saltedpath, f"equirepr_{saltedname}")

    if saltedtype == "density" or saltedtype=="ghost-density":

        # compute rkhs projector and save
        features = h5py.File(os.path.join(sdir,f"FEAT_M-{Menv}.h5"),'r')
        h5f = h5py.File(os.path.join(sdir,  f"projector_M{Menv}_zeta{zeta}.h5"), 'w')
        for spe in species:
            power_env_sparse = features['sparse_descriptors'][spe]['0'][:]
            Mspe = power_env_sparse.shape[0]
            kernel0_mm = np.dot(power_env_sparse,power_env_sparse.T)
            k0 = kernel0_mm**zeta
            V = rkhs_proj(k0)
            h5f.create_dataset(f"projectors/{spe}/0",data=V)
            for lam in range(1,lmax[spe]+1):
                power_env_sparse = features['sparse_descriptors'][spe][str(lam)][:]
                kernel_mm = np.dot(power_env_sparse,power_env_sparse.T)
                for i1 in range(Mspe):
                    for i2 in range(Mspe):
                        kernel_mm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_mm[i1,i2]**(zeta-1)
                V = rkhs_proj(kernel_mm)
                h5f.create_dataset(f"projectors/{spe}/{lam}",data=V)
        h5f.close()
        features.close()

    elif saltedtype == "density-response":

        dirpath = os.path.join(saltedpath, f"normfacts_{saltedname}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        dirpath = os.path.join(saltedpath, f"normfacts_{saltedname}", f"M{Menv}_zeta{zeta}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # compute rkhs projector and save
        features = h5py.File(os.path.join(sdir,f"FEAT_M-{Menv}.h5"),'r')
        features_antisymm = h5py.File(os.path.join(sdir,f"FEAT_M-{Menv}_antisymm.h5"),'r')
        h5f = h5py.File(os.path.join(sdir,  f"projector-response_M{Menv}_zeta{zeta}.h5"), 'w')

        for spe in species:

            print("spe =", spe)

            power_env_sparse = features['sparse_descriptors'][spe]['0'][:]
            Mspe = power_env_sparse.shape[0]
            kernel0_mm = np.dot(power_env_sparse,power_env_sparse.T)

            Mcut = {}
            Mcutsize = {}
            for lam in range(lmax[spe]+1):
                frac = np.exp(-0.05*lam**2)
                Mcut[lam] = int(round(Mspe*frac))
                Mcutsize[lam] = Mcut[lam]*3*(2*lam+1)
                print("lam=",lam,"Mcut=",Mcut[lam],"Msize=",Mcutsize[lam])
            #sys.exit(0)

            for lam in range(lmax[spe]+1):

                print("lam =", lam)

                if lam==0:

                    L = 1
                    print("L=", L)
                    power_env_sparse = features['sparse_descriptors'][spe][str(L)][:]
                    kernel_mm = np.dot(power_env_sparse,power_env_sparse.T)
                    normfact = np.zeros(Mcut[lam]) 
                    for i1 in range(Mcut[lam]):
                        for i2 in range(Mcut[lam]):
                            kernel_mm[i1*3:i1*3+3][:,i2*3:i2*3+3] *= kernel0_mm[i1,i2]**(zeta-1)
                        normfact[i1] = np.sqrt(np.sum(kernel_mm[i1*3:i1*3+3][:,i1*3:i1*3+3]**2))
                    np.save(os.path.join(saltedpath, f"normfacts_{saltedname}", f"M{Menv}_zeta{zeta}", f"normfact_spe-{spe}_lam-{lam}.npy"), normfact)

                    j1 = 0
                    for i1 in range(Mcut[lam]):
                        norm1 = normfact[i1]
                        for imu1 in range(3):
                            j2 = 0
                            for i2 in range(Mcut[lam]):
                                norm2 = normfact[i2]
                                for imu2 in range(3):
                                    kernel_mm[j1,j2] /= np.sqrt(norm1*norm2)
                                    j2 += 1
                            j1 += 1

                    V = rkhs_proj(kernel_mm)
                    h5f.create_dataset(f"projectors/{spe}/{lam}",data=V)

                else:

                    Msize =  Mspe*3*(2*lam+1)
                    kernel_mm = np.zeros((Msize,Msize),complex)                   

                    # Perform CG combination
                    for L in [lam-1,lam,lam+1]:

                        if L==lam:
                            pimag = features_antisymm['sparse_descriptors'][spe][str(L)][:]
                            featsize = pimag.shape[-1]
                            pimag = pimag.reshape(Mspe,2*L+1,featsize)
                            pimag = np.transpose(pimag,(1,0,2)).reshape(2*L+1,Mspe*featsize)
                            preal = np.zeros_like(pimag)
                        else:
                            preal = features['sparse_descriptors'][spe][str(L)][:]
                            featsize = preal.shape[-1]
                            preal = preal.reshape(Mspe,2*L+1,featsize)
                            preal = np.transpose(preal,(1,0,2)).reshape(2*L+1,Mspe*featsize)
                            pimag = np.zeros_like(preal)
                        
                        ptemp = preal + 1j * pimag
                        c2r = sph_utils.complex_to_real_transformation([2*L+1])[0]
                        pcmplx = np.dot(np.conj(c2r.T),ptemp).reshape(2*L+1,Mspe,featsize)
                        pcmplx = np.transpose(pcmplx,(1,0,2)).reshape(Mspe*(2*L+1),featsize)
                        kmm = np.dot(pcmplx,np.conj(pcmplx).T)

                        print("L=", L)
                        # Load the relevant CG coefficients 
                        cgcoefs = np.loadtxt(os.path.join(saltedpath, "wigners", f"cg_response_lam-{lam}_L-{L}.dat"))

                        k0 = kernel0_mm**(zeta-1)
                        cgkernel = kernelequicomb.kernelequicomb(Mspe,Mspe,lam,1,L,Msize,Msize,len(cgcoefs),cgcoefs,kmm.T,k0.T)
                        kernel_mm += cgkernel.T
                                     
                    kernel_mm = kernel_mm[:Mcutsize[lam]][:,:Mcutsize[lam]]

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

                    ktemp1 = np.dot(c2r,np.transpose(kernel_mm.reshape(Mcut[lam],3*(2*lam+1),Mcutsize[lam]),(1,0,2)).reshape(3*(2*lam+1),Mcut[lam]*Mcutsize[lam]))
                    ktemp2 = np.transpose(ktemp1.reshape(3*(2*lam+1),Mcut[lam],Mcutsize[lam]),(1,0,2)).reshape(Mcutsize[lam],Mcutsize[lam]) 
                    kernel_mm = np.dot(ktemp2.reshape(Mcutsize[lam],Mcut[lam],3*(2*lam+1)).reshape(Mcutsize[lam]*Mcut[lam],3*(2*lam+1)),np.conj(c2r).T).reshape(Mcutsize[lam],Mcut[lam],3*(2*lam+1)).reshape(Mcutsize[lam],Mcutsize[lam])
                    
                    normfact = np.zeros(Mcut[lam])
                    for i1 in range(Mcut[lam]):
                        normfact[i1] = np.sqrt(np.sum(np.real(kernel_mm)[i1*3*(2*lam+1):i1*3*(2*lam+1)+3*(2*lam+1)][:,i1*3*(2*lam+1):i1*3*(2*lam+1)+3*(2*lam+1)]**2))
                    np.save(os.path.join(saltedpath, f"normfacts_{saltedname}", f"M{Menv}_zeta{zeta}", f"normfact_spe-{spe}_lam-{lam}.npy"), normfact)

                    knorm = kernelnorm.kernelnorm(Mcut[lam],Mcut[lam],3*(2*lam+1),normfact,normfact,np.real(kernel_mm).T)
                    kernel_mm = knorm.T
                    
                    V = rkhs_proj(kernel_mm)
                    h5f.create_dataset(f"projectors/{spe}/{lam}",data=V)

        h5f.close()
        features.close()
        features_antisymm.close()

if __name__ == "__main__":
    build()
