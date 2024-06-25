"""
Calculate RKHS vectors
"""

import os
import os.path as osp
import time
from typing import Dict, List, Tuple

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from metatensor import Labels
from rascaline import LodeSphericalExpansion, SphericalExpansion
from scipy import sparse

from salted import sph_utils
from salted.lib import equicomb, equicombsparse
from salted.sys_utils import ParseConfig, get_atom_idx, get_conf_range, get_feats_projs, read_system


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
    gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()

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

    if (rank == 0):
        dirpath = os.path.join(saltedpath, fdir, f"M{Menv}_zeta{zeta}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    if size > 1:  comm.Barrier()

    # Distribute structures to tasks
    if parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = range(ndata)

    frames = read(filename,":")

    for iconf in conf_range:

        start_time = time.time()
        print(f"{iconf} start", flush=True)

        structure = frames[iconf]

        # load reference QM data to total array size
        coefs = np.load(osp.join(saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"))
        Tsize = len(coefs)

        if rep1=="rho":
            # get SPH expansion for atomic density
            calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

        elif rep1=="V":
            # get SPH expansion for atomic potential
            calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

        else:
            if rank == 0: print("Error: requested representation", rep1, "not provided")

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

        spx = calculator.compute(structure, selected_keys=keys_selection)
        spx = spx.keys_to_properties("neighbor_type")
        spx = spx.keys_to_samples("center_type")

        # Get 1st set of coefficients as a complex numpy array
        omega1 = np.zeros((nang1+1,natoms[iconf],2*nang1+1,nspe1*nrad1),complex)
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
            if rank == 0: print("Error: requested representation", rep2, "not provided")

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

        spx_pot = calculator.compute(structure, selected_keys=keys_selection)
        spx_pot = spx_pot.keys_to_properties("neighbor_type")
        spx_pot = spx_pot.keys_to_samples("center_type")

        # Get 2nd set of coefficients as a complex numpy array 
        omega2 = np.zeros((nang2+1,natoms[iconf],2*nang2+1,nspe2*nrad2),complex)
        for l in range(nang2+1):
            c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
            omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(o3_lambda=l).values)

        # Compute equivariant features for the given structure
        power = {}
        for lam in range(lmax_max+1):

            [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

            # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
            wigner3j = np.loadtxt(os.path.join(
                saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
            ))
            wigdim = wigner3j.size

            # Reshape arrays of expansion coefficients for optimal Fortran indexing
            v1 = np.transpose(omega1,(2,0,3,1))
            v2 = np.transpose(omega2,(2,0,3,1))

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

        # Compute kernels and projected RKHS features
        Psi:Dict[Tuple[int, str], np.ndarray] = {}
        ispe = {}
        for spe in species:
            ispe[spe] = 0

            # l=0
            if zeta == 1:
                # sparse power spectrum already projected on truncated RKHS
                kernel0_nm = np.dot(power[0][atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                Psi[(spe,0)] = kernel0_nm

            else:

                kernel0_nm = np.dot(power[0][atom_idx[(iconf,spe)]],power_env_sparse[(0,spe)].T)
                kernel_nm = kernel0_nm**zeta
                Psi[(spe,0)] = np.real(np.dot(kernel_nm,Vmat[(0,spe)]))

            # l>0
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


        # build sparse feature-vector memory efficiently
        nrows = Tsize
        ncols = totsize
        srows = arraylist()
        scols = arraylist()
        psi_nonzero = arraylist()
        # srows:List[np.ndarray] = []
        # scols:List[np.ndarray] = []
        # psi_nonzero:List[np.ndarray] = []
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe]+1):
                i1 = ispe[spe]*(2*l+1)
                i2 = ispe[spe]*(2*l+1) + 2*l+1
                x = Psi[(spe,l)][i1:i2]  # 2d array
                nz = np.nonzero(x)  # rwo 0: non-zero row indices, row 1: non-zero column indices
                # vals = x[x!=0]
                vals = x[nz]  # 1d array
                for n in range(nmax[(spe,l)]):
                    psi_nonzero.update(vals)
                    srows.update(nz[0]+i)
                    scols.update(nz[1]+cuml_Mcut[(spe,l,n)])
                    # psi_nonzero.append(vals)
                    # srows.append(nz[0] + i)
                    # scols.append(nz[1] + cuml_Mcut[(spe,l,n)])
                    i += 2*l+1
            ispe[spe] += 1

        psi_nonzero = psi_nonzero.finalize()
        srows = srows.finalize()
        scols = scols.finalize()
        ij = np.vstack((srows,scols))
        # psi_nonzero = np.concatenate(psi_nonzero, axis=0)
        # ij = np.vstack((
        #     np.concatenate(srows, axis=0),
        #     np.concatenate(scols, axis=0)
        # ))

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



if __name__ == "__main__":
    build()
