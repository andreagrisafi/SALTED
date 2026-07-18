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

from salted.sph_utils import equicombfps
from salted.sys_utils import ParseConfig, build_featomic_hyper_params, do_fps, get_atom_idx, read_system

def build():

    inp = ParseConfig().parse_input()
    # frequently used parameters
    saltedpath = inp.salted.saltedpath
    rep1, rep2 = inp.descriptor.rep1.type, inp.descriptor.rep2.type
    nrad1, nrad2 = inp.descriptor.rep1.nrad, inp.descriptor.rep2.nrad
    nang1, nang2 = inp.descriptor.rep1.nang, inp.descriptor.rep2.nang
    neighspe1, neighspe2 = inp.descriptor.rep1.neighspe, inp.descriptor.rep2.neighspe
    nspe1 = len(inp.descriptor.rep1.neighspe)
    nspe2 = len(inp.descriptor.rep2.neighspe)
    ncut = inp.descriptor.sparsify.ncut
    sparsify = ncut > 0
    HP1 = build_featomic_hyper_params(inp.descriptor.rep1)
    HP2 = build_featomic_hyper_params(inp.descriptor.rep2)

    # Generate directories for saving descriptors
    sdir = osp.join(saltedpath, f"equirepr_{inp.salted.saltedname}")
    if not osp.exists(sdir):
        os.mkdir(sdir)

    if not sparsify:
        print(
            "ERROR: inp parameter sparsify=False. "
            "Make sure to include a sparsify section with ncut>0 if you want to sparsify the descriptor\n",
            file=sys.stderr
        )
        sys.exit(1)

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    start = time.time()

    ndata_true = ndata
    print(f"The dataset contains {ndata_true} frames.")

    conf_range = list(range(ndata_true))
    random.Random(3).shuffle(conf_range)

    if inp.descriptor.sparsify.nsamples <= ndata:
        ndata = inp.descriptor.sparsify.nsamples
    else:
        print("ERROR: nsamples cannot be greater than ndata!")
        sys.exit(1)

    conf_range = conf_range[:ndata]
    print(f"Selected {ndata} frames.")

    frames = read(inp.system.filename,":")
    frames = list( frames[i] for i in conf_range )
    natoms = list( natoms[i] for i in conf_range )
    natoms_total = sum(natoms)

    omega1 = sph_utils.get_representation_coeffs(frames, rep1, HP1, 0, neighspe1, species, nang1, nrad1, natoms_total)
    if sph_utils.reps_equivalent(rep1, neighspe1, HP1, rep2, neighspe2, HP2):
        omega2 = omega1
    else:
        omega2 = sph_utils.get_representation_coeffs(frames, rep2, HP2, 0, neighspe2, species, nang2, nrad2, natoms_total)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(1,3,0,2)).copy()
    v2 = np.transpose(omega2,(1,3,0,2)).copy()

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

        print(f"lambda = {lam}, feature space size = {featsize}")

        # Do feature selection with FPS sparsification
        if ncut >= featsize:
            print(f"ERROR: requested number of sparse features larger than total feature space size: {ncut} > {featsize}. Please remove the inp.descriptor.sparsify section or reduce ncut value.")
            sys.exit(1)
        
        pvec = equicombfps(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigner3j,llmax,llvec,lam,c2r,featsize)
        vfps = do_fps(pvec,ncut,verbose=inp.salted.verbose)
        np.save(osp.join(sdir, f"fps{ncut}-{lam}.npy"), vfps)

if __name__ == "__main__":
    build()
