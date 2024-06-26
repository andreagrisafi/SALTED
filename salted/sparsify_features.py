import os
import random
import sys
import time
import os.path as osp

import numpy as np
import h5py
from ase.data import atomic_numbers
from ase.io import read

from salted import sph_utils
from salted import basis

from salted.lib import equicomb, antiequicomb
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

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
    gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()

    nspe1 = len(neighspe1)
    nspe2 = len(neighspe2)

    # Generate directories for saving descriptors
    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")
    if not osp.exists(sdir):
        os.mkdir(sdir)

    if not sparsify:
        print(
            "ERROR: inp parameter sparsify=False. "
            "Make sure to include a sparsify section with ncut>0 if you want to sparsify the descriptor\n",
            file=sys.stderr
        )
        sys.exit(1)

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    start = time.time()

    ndata_true = ndata
    print(f"The dataset contains {ndata_true} frames.")

    conf_range = list(range(ndata_true))
    random.Random(3).shuffle(conf_range)

    if nsamples <= ndata:
        ndata = nsamples
    else:
        print("ERROR: nsamples cannot be greater than ndata!")
        sys.exit(1)

    conf_range = conf_range[:ndata]
    print(f"Selected {ndata} frames.")

    frames = read(filename,":")
    frames = list( frames[i] for i in conf_range )
    natoms = list( natoms[i] for i in conf_range )
    natoms_total = sum(natoms)

    def do_fps(x, d=0, initial=-1):
        # Code from Giulio Imbalzano

        if d == 0 : d = len(x)
        n = len(x)
        iy = np.zeros(d,int)
        if (initial == -1):
            iy[0] = np.random.randint(0,n)
        else:
            iy[0] = initial
        # Faster evaluation of Euclidean distance
        # Here we fill the n2 array in this way because it halves the memory cost of this routine
        n2 = np.array([np.sum(x[i] * np.conj([x[i]])) for i in range(len(x))])
        dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
        for i in range(1,d):
            print("Doing ",i," of ",d," dist = ",max(dl))
            iy[i] = np.argmax(dl)
            nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
            dl = np.minimum(dl,nd)
        return iy

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

    omega1 = sph_utils.get_representation_coeffs(frames,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe1,species,nang1,nrad1,natoms_total)
    omega2 = sph_utils.get_representation_coeffs(frames,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe2,species,nang2,nrad2,natoms_total)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    for lam in range(lmax_max+1):

        print("lambda =", lam)

        [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(osp.join(saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"))
        wigdim = wigner3j.size

        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

        # compute normalized equivariant descriptor
        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
        p = np.transpose(p,(2,0,1))

        print(f"feature space size = {featsize}")

        #TODO modify SALTED to directly deal with compact natoms_total dimension
        if lam==0:
            p = p.reshape(natoms_total,featsize)
            pvec = np.zeros((ndata,natmax,featsize))
        else:
            p = p.reshape(natoms_total,2*lam+1,featsize)
            pvec = np.zeros((ndata,natmax,2*lam+1,featsize))

        j = 0
        for i in range(ndata):
            for iat in range(natoms[i]):
                pvec[i,iat] = p[j]
                j += 1

        # Do feature selection with FPS sparsification
        if ncut >= featsize:
            ncut = featsize

        print("fps...")
        pvec = pvec.reshape(ndata*natmax*(2*lam+1),featsize)
        vfps = do_fps(pvec.T,ncut,0)
        np.save(osp.join(sdir, f"fps{ncut}-{lam}.npy"), vfps)

        if saltedtype=="density-response" and lam>0 and lam<lmax_max:

            [llmax,llvec] = sph_utils.get_angular_indexes_antisymmetric(lam,nang1,nang2)

            # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
            wigner3j = np.loadtxt(osp.join(saltedpath, "wigners", f"wigner_antisymm_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"))
            wigdim = wigner3j.size

            # Compute complex to real transformation matrix for the given lambda value
            c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

            # compute normalized equivariant descriptor
            featsize = nspe1*nspe2*nrad1*nrad2*llmax
            p = antiequicomb.antiequicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
            p = np.transpose(p,(2,0,1))

            print(f"feature space size = {featsize}")

            p = p.reshape(natoms_total,2*lam+1,featsize)
            pvec = np.zeros((ndata,natmax,2*lam+1,featsize))

            j = 0
            for i in range(ndata):
                for iat in range(natoms[i]):
                    pvec[i,iat] = p[j]
                    j += 1

            # Do feature selection with FPS sparsification
            if ncut >= featsize:
                ncut = featsize

            print("fps...")
            pvec = pvec.reshape(ndata*natmax*(2*lam+1),featsize)
            vfps = do_fps(pvec.T,ncut,0)
            np.save(osp.join(sdir, f"fps{ncut}-{lam}_antisymm.npy"), vfps)

    print(f"time: {(time.time()-start):.2f}")


if __name__ == "__main__":
    build()
