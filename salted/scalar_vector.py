import os
import sys
import time
import os.path as osp
from ase.io import read
import h5py

import numpy as np
from scipy import sparse
from ase.data import atomic_numbers

from salted.sys_utils import read_system,get_atom_idx,get_conf_range

from salted import sph_utils
from salted import basis
from salted.sys_utils import ParseConfig

from salted.lib import equicomb
from salted.lib import equicombsparse

def build():

    inp = ParseConfig().parse_input()
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

    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")

    if sparsify==False:
        # Generate directories for saving descriptors
        if not osp.exists(sdir):
            os.mkdir(sdir)

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # Load feature space sparsification information if required
    if sparsify:
        vfps = {}
        for lam in range(lmax_max+1):
            vfps[lam] = np.load(osp.join(
                saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    frames = read(filename,":")
    natoms_total = sum(natoms)
    conf_range = range(ndata)

    lam = 0
    [llmax,llvec] = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

    # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
    wigner3j = np.loadtxt(os.path.join(
        saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
    ))
    wigdim = wigner3j.size

    omega1 = sph_utils.get_representation_coeffs(frames,rep1,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe1,species,nang1,nrad1,natoms_total)
    omega2 = sph_utils.get_representation_coeffs(frames,rep2,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,0,neighspe2,species,nang2,nrad2,natoms_total)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

    # Compute complex to real transformation matrix for the given lambda value
    c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

    start = time.time()

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

    print("time = ", time.time()-start)

    #TODO modify SALTED to directly deal with compact natoms_total dimension
    p = p.reshape(natoms_total,featsize)
    pvec = np.zeros((ndata,natmax,featsize))

    j = 0
    for i,iconf in enumerate(conf_range):
        for iat in range(natoms[iconf]):
            pvec[i,iat] = p[j]
            j += 1

    h5f = h5py.File(osp.join(sdir, f"FEAT-0.h5"), 'w')
    h5f.create_dataset("descriptor",data=pvec)
    h5f.close()

if __name__ == "__main__":
    build()
