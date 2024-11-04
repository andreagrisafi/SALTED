import os
import os.path as osp
import sys
import time
from typing import Dict, List

import h5py
import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from scipy import special

from salted import basis, sph_utils
from salted.lib import equicomb, equicombsparse, equicombnonorm, antiequicombnonorm, kernelequicomb
from salted.sys_utils import (
    PLACEHOLDER,
    ParseConfig,
    get_atom_idx,
    get_conf_range,
    get_feats_projs,
    get_feats_projs_response,
    read_system,
)
from salted.cp2k.utils import init_moments, compute_charge_and_dipole, compute_polarizability

def build():

    inp = ParseConfig().parse_input()
    (saltedname, saltedpath, saltedtype,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    if filename_pred == PLACEHOLDER or predname == PLACEHOLDER:
        raise ValueError(
            "No prediction file and name provided, "
            "please specify the entry named `prediction.filename` and `prediction.predname` in the input file."
        )

    if parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
        size = 1

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(filename_pred, species, dfbasis)
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    bohr2angs = 0.529177210670

    # Distribute structures to tasks
    ndata_true = ndata
    if rank == 0: print(f"The dataset contains {ndata_true} frames.")
    if parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)  # List[int]
        ndata = len(conf_range)
        natmax = max(natoms[conf_range])
        print(f"Task {rank+1} handles the following structures: {conf_range}", flush=True)
    else:
        conf_range = list(range(ndata))
    natoms_total = sum(natoms[conf_range])

    reg_log10_intstr = str(int(np.log10(regul)))  # for consistency

    # load regression weights
    ntrain = int(Ntrain * trainfrac)
    weights = np.load(osp.join(
        saltedpath,
        f"regrdir_{saltedname}",
        f"M{Menv}_zeta{zeta}",
        f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    if qmcode=="cp2k":
        # Initialize calculation of density/density-response moments
        alphas,sigmas,charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,rank)

    # base directory path for this prediction
    dirpath = osp.join(
        saltedpath,
        f"predictions_{saltedname}_{predname}",
        f"M{Menv}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}",
    )

    # Create directory for predictions
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if saltedtype=="density-response":
            for icart in ["x","y","z"]:
                cartpath = os.path.join(dirpath, f"{icart}")
                if not os.path.exists(cartpath):
                    os.mkdir(cartpath)
    if size > 1: comm.Barrier()

    # Initialize files for derived properties 
    if qmcode=="cp2k":
        if saltedtype=="density":
            qfile = init_property_file("charges",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
            dfile = init_property_file("dipoles",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
        if saltedtype=="density-response":
            pfile = init_property_file("polarizabilities",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)

    start = time.time()

    # Read frames
    frames = read(filename_pred,":")
    frames = [frames[i] for i in conf_range]

    # Compute atom-density spherical expansion coefficients
    omega1 = sph_utils.get_representation_coeffs(frames,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe1,species,nang1,nrad1,natoms_total)
    omega2 = sph_utils.get_representation_coeffs(frames,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe2,species,nang2,nrad2,natoms_total)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

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

        # Load spherical averages if required
        if average:
            av_coefs = {}
            for spe in species:
                av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

        # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
        pvec = {}
        for lam in range(lmax_max+1):

            if rank == 0: print(f"lambda = {lam}")

            llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

            # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
            wigner3j = np.loadtxt(osp.join(
                saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
            ))
            wigdim = wigner3j.size

            # Compute complex to real transformation matrix for the given lambda value
            c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

            if sparsify:

                featsize = nspe1*nspe2*nrad1*nrad2*llmax
                nfps = len(vfps[lam])
                p = equicombsparse.equicombsparse(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize,nfps,vfps[lam])
                p = np.transpose(p,(2,0,1))
                featsize = ncut

            else:

                featsize = nspe1*nspe2*nrad1*nrad2*llmax
                p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
                p = np.transpose(p,(2,0,1))

            # Fill vector of equivariant descriptor
            if lam==0:
                p = p.reshape(natoms_total,featsize)
                pvec[lam] = np.zeros((ndata,natmax,featsize))
            else:
                p = p.reshape(natoms_total,2*lam+1,featsize)
                pvec[lam] = np.zeros((ndata,natmax,2*lam+1,featsize))

            j = 0
            for i,iconf in enumerate(conf_range):
                for iat in range(natoms[iconf]):
                    pvec[lam][i,iat] = p[j]
                    j += 1

        """ save descriptor of the prediction dataset """
        if inp.prediction.save_descriptor:
            if rank == 0:
                print(f"Saving descriptor of the prediction dataset to dir {dirpath}")
            save_pred_descriptor(pvec, conf_range, list(natoms[conf_range]), dirpath)

        psi_nm = {}
        for i,iconf in enumerate(conf_range):

            Tsize = 0
            for iat in range(natoms[iconf]):
                spe = atomic_symbols[iconf][iat]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        Tsize += 2*l+1

            for spe in species:

                # lam = 0
                if zeta==1:
                    psi_nm[(spe,0)] = np.dot(pvec[0][i,atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                else:
                    kernel0_nm = np.dot(pvec[0][i,atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                    kernel_nm = kernel0_nm**zeta
                    psi_nm[(spe,0)] = np.dot(kernel_nm,Vmat[(0,spe)])

                # lam > 0
                for lam in range(1,lmax[spe]+1):

                    featsize = pvec[lam].shape[-1]
                    if zeta==1:
                        psi_nm[(spe,lam)] = np.dot(pvec[lam][i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                    else:
                        kernel_nm = np.dot(pvec[lam][i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                        for i1 in range(natom_dict[(iconf,spe)]):
                            for i2 in range(Mspe[spe]):
                                kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[i1,i2]**(zeta-1)
                        psi_nm[(spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])

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
            i = 0
            pred_coefs = np.zeros(Tsize)
            for iat in range(natoms[iconf]):
                spe = atomic_symbols[iconf][iat]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                        if average and l==0:
                            Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1
                ispe[spe] += 1

            # add back spherical averages if required
            if average:
                pred_coefs += Av_coeffs

            # save predicted coefficients 
            np.savetxt(osp.join(dirpath, f"COEFFS-{iconf+1}.dat"), pred_coefs)

            if qmcode=="cp2k":
                # Compute charges and dipole moments
                charge, dipole = compute_charge_and_dipole(frames[iconf],inp.qm.pseudocharge,natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs,average)
                print(iconf+1,charge,file=qfile)
                print(iconf+1,dipole["x"],dipole["y"],dipole["z"],file=dfile)
    
    elif saltedtype=="density-response":

        # Load training feature vectors and RKHS projection matrix
        Vmat,Mspe,power_env_sparse,power_env_sparse_antisymm = get_feats_projs_response(species,lmax)

        lmax_max += 1

        cart = ["y","z","x"]

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
            p = equicombnonorm.equicombnonorm(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
            p = np.transpose(p,(2,0,1))

            # Fill vector of equivariant descriptor
            if lam==0:
                p = p.reshape(natoms_total,featsize)
                power[lam] = np.zeros((ndata,natmax,featsize))
            else:
                p = p.reshape(natoms_total,2*lam+1,featsize)
                power[lam] = np.zeros((ndata,natmax,2*lam+1,featsize))

            j = 0
            for i,iconf in enumerate(conf_range):
                for iat in range(natoms[iconf]):
                    power[lam][i,iat] = p[j]
                    j += 1

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
            p = antiequicombnonorm.antiequicombnonorm(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
            p = np.transpose(p,(2,0,1))

            # Fill vector of equivariant descriptor
            p = p.reshape(natoms_total,2*lam+1,featsize)

            power_antisymm[lam] = np.zeros((ndata,natmax,2*lam+1,featsize))
            j = 0
            for i,iconf in enumerate(conf_range):
                for iat in range(natoms[iconf]):
                    power_antisymm[lam][i,iat] = p[j]
                    j += 1

        psi_nm = {}
        psi_nm_cart = {}
        for i,iconf in enumerate(conf_range):

            Tsize = 0
            for iat in range(natoms[iconf]):
                spe = atomic_symbols[iconf][iat]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        Tsize += 2*l+1

            # Compute kernels and RKHS descriptors
            for ic in cart:
                for spe in species:
                    for lam in range(lmax[spe]+1):
                        psi_nm_cart[(ic,spe,lam)] = np.zeros((natom_dict[(iconf,spe)]*(2*lam+1),Vmat[(lam,spe)].shape[-1]))

            for spe in species:

                Mcut = {}
                Mcutsize = {}
                for lam in range(lmax[spe]+1):
                    frac = np.exp(-0.05*lam**2)
                    Mcut[lam] = int(round(Mspe[spe]*frac))
                    Mcutsize[lam] = Mcut[lam]*3*(2*lam+1)

                # lam=0
                kernel0_nm = np.dot(power[0][i,atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                kernel_nm = np.dot(power[1][i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*3,power[1].shape[-1]),power_env_sparse[(1,spe)].T)
                for i1 in range(natom_dict[(iconf,spe)]):
                    for i2 in range(Mspe[spe]):
                        kernel_nm[i1*3:i1*3+3][:,i2*3:i2*3+3] *= kernel0_nm[i1,i2]**(zeta-1)
                kernel_nm = kernel_nm[:,:Mcutsize[0]]

                kernel0_nn = np.dot(power[0][i,atom_idx[(iconf,spe)]],power[0][i,atom_idx[(iconf,spe)]].T)
                kernel_nn = np.dot(power[1][i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*3,power[1].shape[-1]),power[1][i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*3,power[1].shape[-1]).T)
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
                        for i2 in range(Mcut[0]):
                            norm2 = normfact_sparse[i2]
                            for imu2 in range(3):
                                kernel_nm[j1,j2] /= np.sqrt(norm1*norm2)
                                j2 += 1
                        j1 += 1

                psi_nm[(spe,0)] = np.real(np.dot(kernel_nm,Vmat[(0,spe)]))

                idx = 0
                idx_cart = 0
                for iat in range(natom_dict[(iconf,spe)]):
                    for ik in range(3):
                        psi_nm_cart[(cart[ik],spe,0)][idx_cart] = psi_nm[(spe,0)][idx]
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
                            pimag = power_antisymm[L][i,atom_idx[(iconf,spe)]]
                            featsize = pimag.shape[-1]
                            pimag = pimag.reshape(natom_dict[(iconf,spe)],2*L+1,featsize)
                            pimag = np.transpose(pimag,(1,0,2)).reshape(2*L+1,natom_dict[(iconf,spe)]*featsize)
                            preal = np.zeros_like(pimag)
                        else:
                            preal = power[L][i,atom_idx[(iconf,spe)]]
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

                    kernel_nm = kernel_nm[:,:Mcutsize[lam]]

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
                    ktemp1 = np.dot(c2r,np.transpose(kernel_nm.reshape(natom_dict[(iconf,spe)],3*(2*lam+1),Mcutsize[lam]),(1,0,2)).reshape(3*(2*lam+1),natom_dict[(iconf,spe)]*Mcutsize[lam]))
                    ktemp2 = np.transpose(ktemp1.reshape(3*(2*lam+1),natom_dict[(iconf,spe)],Mcutsize[lam]),(1,0,2)).reshape(Nsize,Mcutsize[lam])
                    kernel_nm = np.dot(ktemp2.reshape(Nsize,Mcut[lam],3*(2*lam+1)).reshape(Nsize*Mcut[lam],3*(2*lam+1)),np.conj(c2r).T).reshape(Nsize,Mcut[lam],3*(2*lam+1)).reshape(Nsize,Mcutsize[lam])

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
                            for i2 in range(Mcut[lam]):
                                norm2 = normfact_sparse[i2]
                                for imu2 in range(3*(2*lam+1)):
                                    kernel_nm[j1,j2] /= np.sqrt(norm1*norm2)
                                    j2 += 1
                            j1 += 1

                    # project kernel on the RKHS
                    psi_nm[(spe,lam)] = np.real(np.dot(np.real(kernel_nm),Vmat[(lam,spe)]))

                    idx = 0
                    idx_cart = 0
                    for iat in range(natom_dict[(iconf,spe)]):
                        for imu in range(2*lam+1):
                            for ik in range(3):
                                psi_nm_cart[(cart[ik],spe,lam)][idx_cart] = psi_nm[(spe,lam)][idx]
                                idx += 1
                            idx_cart += 1

            pred_coefs = {}
            for icart in ["x","y","z"]:
                
                # compute predictions per channel
                C = {}
                ispe = {}
                isize = 0
                for spe in species:
                    ispe[spe] = 0
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            Mcut = psi_nm_cart[(icart,spe,l)].shape[1]
                            C[(spe,l,n)] = np.dot(psi_nm_cart[(icart,spe,l)],weights[isize:isize+Mcut])
                            isize += Mcut

                # fill vector of predictions
                i = 0
                pred_coefs[icart] = np.zeros(Tsize)
                for iat in range(natoms[iconf]):
                    spe = atomic_symbols[iconf][iat]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            pred_coefs[icart][i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                            i += 2*l+1
                    ispe[spe] += 1

                # save predicted coefficients 
                np.savetxt(osp.join(dirpath, f"{icart}", f"COEFFS-{iconf+1}.dat"), pred_coefs[icart])

            if qmcode=="cp2k":
                # Compute polarizability
                alpha = compute_polarizability(frames[iconf],natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs)

                # Save polarizabilities
                print(iconf+1, alpha[("x","x")],    alpha[("x","y")],    alpha[("x","z")],
                               alpha[("y","x")],    alpha[("y","y")],    alpha[("y","z")],
                               alpha[("z","x")],    alpha[("z","y")],    alpha[("z","z")],
                               file=pfile)

    if qmcode == "cp2k":
        if saltedtype=="density":
            qfile.close()
            dfile.close()
        if saltedtype=="density-response":
            pfile.close()

    if rank == 0: print(f"\ntotal time: {(time.time()-start):.2f} s")


def save_pred_descriptor(data:Dict[int, np.ndarray], config_range:List[int], natoms:List[int], dpath:str):
    """Save the descriptor data of the prediction dataset.

    Args:
        data (Dict[int, np.ndarray]): the descriptor data to be saved.
            int -> lambda value,
            np.ndarray -> descriptor data, shape (ndata, natmax, [2*lambda+1,] featsize)
                natmax should be cut to the number of atoms in the structure (natoms[i])
                2*lambda+1 is only for lambda > 0.
        config_range (List[int]): the indices of the structures in the full dataset.
        natoms (List[int]): the number of atoms in each structure. Should be the same length as config_range.
        dpath (str): the directory to save the descriptor data.

    Output:
        The descriptor data of each structure is saved in a separate npz file in the directory dpath named as
        "descriptor_{i}.npz", where i starts from 1.
        Format: npz file with keys as lambda values and values as the descriptor data.
            Values have shape (natom, [2*lambda+1,] featsize). 2*lambda+1 is only for lambda > 0.
    """
    assert len(config_range) == len(natoms), f"The length of config_range and natoms should be the same, " \
        f"but get {config_range=} and {natoms=}."
    for lam, data_this_lam in data.items():
        assert data_this_lam.shape[0] == len(config_range), \
            f"The first dimension of the descriptor data should be the same as the length of config_range, " \
            f"but at {lam=} get {data_this_lam.shape[0]=} and {len(config_range)=}."

    """ cut natmax to the number of atoms in the structure """
    for idx, idx_in_full_dataset in enumerate(config_range):
        this_data:Dict[int, np.ndarray] = dict()
        this_natoms = natoms[idx]
        for lam, data_this_lam in data.items():
            this_data[f"lam{lam}"] = data_this_lam[idx, :this_natoms]  # shape (natom, [2*lambda+1,] featsize)
        with open(osp.join(dpath, f"descriptor_{idx_in_full_dataset+1}.npz"), "wb") as f:  # index starts from 1
            np.savez(f, **this_data)


if __name__ == "__main__":
    build()
