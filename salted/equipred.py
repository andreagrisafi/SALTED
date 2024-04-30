import os
import os.path as osp
import sys
import time

import h5py
import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from metatensor import Labels
from rascaline import LodeSphericalExpansion, SphericalExpansion
from scipy import special

from salted import basis, sph_utils
from salted.lib import equicomb, equicombsparse
from salted.sys_utils import (
    PLACEHOLDER,
    ParseConfig,
    get_atom_idx,
    get_conf_range,
    read_system,
)


def build():

    inp = ParseConfig().parse_input()
    (saltedname, saltedpath,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()

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
    if parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        ndata = len(conf_range)
        print(f"Task {rank+1} handles the following structures: {conf_range}", flush=True)
    else:
        conf_range = list(range(ndata))
    natoms_total = sum(natoms[conf_range])

    if qmcode=="cp2k":

        bdir = osp.join(saltedpath,"basis")
        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        if rank == 0: print("Reading auxiliary basis info...")
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                avals = np.loadtxt(osp.join(bdir,f"{spe}-{dfbasis}-alphas-L{l}.dat"))
                if nmax[(spe,l)]==1:
                    alphas[(spe,l,0)] = float(avals)
                    sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
                else:
                    for n in range(nmax[(spe,l)]):
                        alphas[(spe,l,n)] = avals[n]
                        sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

        # compute integrals of basis functions (needed to a posteriori correction of the charge)
        charge_integrals = {}
        dipole_integrals = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                charge_integrals_temp = np.zeros(nmax[(spe,l)])
                dipole_integrals_temp = np.zeros(nmax[(spe,l)])
                for n in range(nmax[(spe,l)]):
                    inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                    charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                    dipole_radint = 2**float(1.0+float(l)/2.0) * sigmas[(spe,l,n)]**(4+l) * special.gamma(2.0+float(l)/2.0)
                    charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
                    dipole_integrals[(spe,l,n)] = dipole_radint * np.sqrt(4.0*np.pi/3.0) / np.sqrt(inner)

    # Load feature space sparsification information if required
    if sparsify:
        vfps = {}
        for lam in range(lmax_max+1):
            vfps[lam] = np.load(osp.join(
                saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    # Load training feature vectors and RKHS projection matrix
    Vmat = {}
    Mspe = {}
    power_env_sparse = {}
    for spe in species:
        for lam in range(lmax[spe]+1):
            # load RKHS projectors
            Vmat[(lam,spe)] = np.load(osp.join(
                saltedpath,
                f"equirepr_{saltedname}",
                f"spe{spe}_l{lam}",
                f"projector_M{Menv}_zeta{zeta}.npy",
            ))
            # load sparse equivariant descriptors
            power_env_sparse[(lam,spe)] = h5py.File(osp.join(
                saltedpath,
                f"equirepr_{saltedname}",
                f"spe{spe}_l{lam}",
                f"FEAT_M-{Menv}.h5"
            ), 'r')['sparse_descriptor'][:]
            if lam == 0:
                Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
            # precompute projection on RKHS if linear model
            if zeta==1:
                power_env_sparse[(lam,spe)] = np.dot(
                    Vmat[(lam,spe)].T, power_env_sparse[(lam,spe)]
                )

    reg_log10_intstr = str(int(np.log10(regul)))  # for consistency

    # load regression weights
    ntrain = int(Ntrain * trainfrac)
    weights = np.load(osp.join(
        saltedpath,
        f"regrdir_{saltedname}",
        f"M{Menv}_zeta{zeta}",
        f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    start = time.time()

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

    frames = read(filename_pred,":")
    frames = [frames[i] for i in conf_range]

    if rank == 0: print(f"The dataset contains {ndata_true} frames.")

    if rep1=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep1=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        if rank == 0:
            raise ValueError(f"Unknown representation {rep1=}, expected 'rho' or 'V'")
        else:
            exit()

    nspe1 = len(neighspe1)
    keys_array = np.zeros(((nang1+1)*len(species)*nspe1,4),int)
    i = 0
    for l in range(nang1+1):
        for specen in species:
            for speneigh in neighspe1:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    spx = calculator.compute(frames, selected_keys=keys_selection)
    spx = spx.keys_to_properties("neighbor_type")
    spx = spx.keys_to_samples("center_type")

    # Get 1st set of coefficients as a complex numpy array
    omega1 = np.zeros((nang1+1,natoms_total,2*nang1+1,nspe1*nrad1),complex)
    for l in range(nang1+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(o3_lambda=l).values)

    if rep2=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep2=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        if rank == 0:
            raise ValueError(f"Unknown representation {rep2=}, expected 'rho' or 'V'")
        else:
            exit()

    nspe2 = len(neighspe2)
    keys_array = np.zeros(((nang2+1)*len(species)*nspe2,4),int)
    i = 0
    for l in range(nang2+1):
        for specen in species:
            for speneigh in neighspe2:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    spx_pot = calculator.compute(frames, selected_keys=keys_selection)
    spx_pot = spx_pot.keys_to_properties("neighbor_type")
    spx_pot = spx_pot.keys_to_samples("center_type")

    # Get 2nd set of coefficients as a complex numpy array
    omega2 = np.zeros((nang2+1,natoms_total,2*nang2+1,nspe2*nrad2),complex)
    for l in range(nang2+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(o3_lambda=l).values)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

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
            os.makedirs(dirpath)
    if size > 1:  comm.Barrier()

    if qmcode=="cp2k":
        xyzfile = read(filename_pred,":")
        q_fpath = osp.join(dirpath, "charges.dat")
        d_fpath = osp.join(dirpath, "dipoles.dat")
        if rank == 0:
            # remove old output files
            remove_if_exists = lambda fpath: os.remove(fpath) if os.path.exists(fpath) else None
            remove_if_exists(q_fpath)
            remove_if_exists(d_fpath)
        if parallel: comm.Barrier()
        qfile = open(q_fpath, "a")
        dfile = open(d_fpath, "a")

    # Load spherical averages if required
    if average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    pvec = {}
    for lam in range(lmax_max+1):

        if rank == 0: print(f"lambda = {lam}")

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

    if parallel:
        comm.Barrier()
        for lam in range(lmax_max+1):
            pvec[lam] = comm.allreduce(pvec[lam])

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

        if qmcode=="cp2k":

            # get geometry ingormation for dipole calculation
            geom = xyzfile[iconf]
            geom.wrap()
            coords = geom.get_positions()/bohr2angs
            cell = geom.get_cell()/bohr2angs
            all_symbols = xyzfile[iconf].get_chemical_symbols()
            all_natoms = int(len(all_symbols))

            # compute integral of predicted density
            iaux = 0
            rho_int = 0.0
            nele = 0.0
            for iat in range(all_natoms):
                spe = all_symbols[iat]
                if spe in species:
                    nele += inp.qm.pseudocharge
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                                rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                            iaux += 2*l+1

            # compute charge and dipole
            iaux = 0
            charge = 0.0
            charge_right = 0.0
            dipole = 0.0
            for iat in range(all_natoms):
                spe = all_symbols[iat]
                if spe in species:
                   if average:
                       dipole += inp.qm.pseudocharge * coords[iat,2]
                   for l in range(lmax[spe]+1):
                       for n in range(nmax[(spe,l)]):
                           for im in range(2*l+1):
                               if l==0 and im==0:
                                   if average:
                                       pred_coefs[iaux] *= nele/rho_int
                                   else:
                                       if n==nmax[(spe,l)]-1:
                                           pred_coefs[iaux] -= rho_int/(charge_integrals[(spe,l,n)]*natoms[iconf])
                                   charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                                   dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                               if l==1 and im==1:
                                   dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)]
                               iaux += 1
            print(iconf+1,dipole,file=dfile)
            print(iconf+1,rho_int,charge,file=qfile)

        # save predicted coefficients 
        np.savetxt(osp.join(dirpath, f"COEFFS-{iconf+1}.dat"), pred_coefs)


    if qmcode=="cp2k":
        qfile.close()
        dfile.close()
        if parallel and rank == 0:
            d_fpath = osp.join(dirpath, "dipoles.dat")
            dips = np.loadtxt(d_fpath)
            np.savetxt(d_fpath, dips[dips[:,0].argsort()], fmt='%i %f')
            q_fpath = osp.join(dirpath, "charges.dat")
            qs = np.loadtxt(q_fpath)
            np.savetxt(q_fpath, qs[qs[:,0].argsort()],fmt='%i %f')

    if rank == 0: print(f"\ntotal time: {(time.time()-start):.2f} s")



if __name__ == "__main__":
    build()
