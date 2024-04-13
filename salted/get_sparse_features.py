"""
TODO: replace class arraylist by numpy.concatenate
"""

import os
import sys
import time
import os.path as osp
from ase.io import read
import h5py

import numpy as np
from scipy import sparse
from ase.data import atomic_numbers

from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from metatensor import Labels

from salted import wigner
from salted import sph_utils
from salted import basis

from salted.lib import equicomb
from salted.lib import equicombsparse

def build():
    inp = ParseConfig().parse_input()

    # salted parameters
    (saltedname, saltedpath,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    z, Menv, Ntrain, trainfrac, regul, eigcut,
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


    # Load training feature vectors and RKHS projection matrix
    vfps = {}
    for lam in range(lmax_max+1):
        # Load sparsification details
        if sparsify:
            vfps[lam] = np.load(osp.join(
                saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")

    # Distribute structures to tasks
    if parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = range(ndata)

    sparse_set = np.loadtxt(osp.join(sdir, f"sparse_set_{Menv}.txt"),int)
    fps_idx = sparse_set[:,0]
    fps_species = sparse_set[:,1]

    Mspe = {}
    for spe in species:
        Mspe[spe] = 0

    fps_indexes_per_conf = {}
    for iconf in range(ndata):
        for spe in species:
            fps_indexes_per_conf[(iconf,spe)] = []

    itot = 0
    Midx_spe = {}
    for iconf in range(ndata):
        for spe in species:
            Midx_spe[(iconf,spe)] = Mspe[spe]
        for iat in range(natoms[iconf]):
            if itot in fps_idx:
                for spe in species:
                    if iat in atom_idx[(iconf,spe)]:
                        fps_indexes_per_conf[(iconf,spe)].append(iat)
                        Mspe[spe] += 1
            itot += 1

    power_env_sparse = {}
    for spe in species:
        for lam in range(lmax_max+1):
            if sparsify:
                featsize = ncut
            else:
                llmax = 0
                for l1 in range(nang1+1):
                    for l2 in range(nang2+1):
                        # keep only even combination to enforce inversion symmetry
                        if (lam+l1+l2)%2==0 :
                            if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                                llmax+=1
                nspe1 = len(neighspe1)
                nspe2 = len(neighspe2)
                featsize = nspe1*nspe2*nrad1*nrad2*llmax
            if lam==0:
                power_env_sparse[(spe,lam)] = np.zeros((Mspe[spe],featsize))
            else:
                power_env_sparse[(spe,lam)] = np.zeros((Mspe[spe],(2*lam+1),featsize))

    frames = read(filename,":")

    for iconf in conf_range:

        start_time = time.time()
        print(f"{iconf} start", flush=True)

        structure = frames[iconf]

        # load reference QM data to total array size
        coefs = np.load(osp.join(saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"))
        Tsize = len(coefs)

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
        for lam in range(lmax_max+1):

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
                power = p.reshape(natoms[iconf],featsize)
            else:
                power = p.reshape(natoms[iconf],2*lam+1,featsize)

            for spe in species:
                nfps = len(fps_indexes_per_conf[(iconf,spe)])
                power_env_sparse[(spe,lam)][Midx_spe[(iconf,spe)]:Midx_spe[(iconf,spe)]+nfps] = power[fps_indexes_per_conf[(iconf,spe)]]

        end_time = time.time()
        print(f"{iconf} end, time cost = {(end_time - start_time):.2f} s", flush=True)

    if parallel:
        comm.Barrier()
        for spe in species:
            for lam in range(lmax[spe]+1):
                power_env_sparse[(spe,lam)] = comm.allreduce(power_env_sparse[(spe,lam)])

    if rank==0:

        # reshape sparse vector and save
        h5f = h5py.File(osp.join(sdir,  f"FEAT_M-{M}.h5"), 'w')
        for spe in species:
            for lam in range(lmax[spe]+1):
                power_env_sparse[(spe,lam)] = power_env_sparse[(spe,lam)].reshape(Mspe[spe]*(2*lam+1),power_env_sparse[(spe,lam)].shape[-1])
                h5f.create_dataset(f"sparse_descriptors/{spe}/{lam}",data=power_env_sparse[(spe,lam)])
        h5f.close()

if __name__ == "__main__":
    build()
