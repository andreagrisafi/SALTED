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

from salted.lib import equicomb, equicombfps
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range, do_fps

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

    omega1 = sph_utils.get_representation_coeffs(frames,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe1,species,nang1,nrad1,natoms_total)
    omega2 = sph_utils.get_representation_coeffs(frames,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe2,species,nang2,nrad2,natoms_total)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    for lam in range(lmax_max+1):

        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(osp.join(saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"))
        wigdim = wigner3j.size

        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

        # compute normalized equivariant descriptor
        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        # p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
        # p = np.transpose(p,(2,0,1))

        # #TODO modify SALTED to directly deal with compact natoms_total dimension
        # if lam==0:
        #     p = p.reshape(natoms_total,featsize)
        #     pvec = np.zeros((ndata,natmax,featsize))
        # else:
        #     p = p.reshape(natoms_total,2*lam+1,featsize)
        #     pvec = np.zeros((ndata,natmax,2*lam+1,featsize))

        # j = 0
        # for i in range(ndata):
        #     for iat in range(natoms[i]):
        #         pvec[i,iat] = p[j]
        #         j += 1

        # Do feature selection with FPS sparsification
        if ncut >= featsize:
            print("ERROR: requested number of sparse features larger than total feature space size! Please get rid of the inp.descriptor.sparsify section.")
            sys.exit(1)

        # print("fps...")
        # pvec = pvec.reshape(ndata*natmax*(2*lam+1),featsize)
        # vfps = do_fps(pvec.T,ncut)
        
        pvec = equicombfps.equicombfps(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
        vfps = do_fps(pvec,ncut)
        np.save(osp.join(sdir, f"fps{ncut}-{lam}.npy"), vfps)


if __name__ == "__main__":
    build()
