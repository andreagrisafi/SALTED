import os
import sys
import time
import os.path as osp
import io

import ase
import numpy as np
from sympy.physics.wigner import wigner_3j

from salted import sph_utils
from salted.sys_utils import ParseConfig

def build():
    inp = ParseConfig().parse_input()

    from salted.sys_utils import read_system, get_atom_idx
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    nang1, nang2 = inp.descriptor.rep1.nang, inp.descriptor.rep2.nang

    # Generate directories for saving descriptors
    dirpath = os.path.join(inp.salted.saltedpath, "wigners")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    def get_wigner3j(llmax:int, llvec:np.ndarray, lam:int, wig:io.TextIOWrapper):
        """Compute and save Wigner-3J symbols needed for symmetry-adapted combination"""

        for il in range(llmax):
            l1 = int(llvec[il,0])
            l2 = int(llvec[il,1])
            for imu in range(2*lam+1):
                mu = imu-lam
                for im1 in range(2*l1+1):
                    m1 = im1-l1
                    m2 = m1-mu
                    if abs(m2) <= l2:
                        im2 = m2+l2
                        # for wigner_3j, all the parameters should be integers or half-integers
                        w3j = wigner_3j(lam,l2,l1,mu,m2,-m1) * (-1.0)**(m1)
                        print(float(w3j),file=wig)

    for lam in range(lmax_max+1):

        print(f"wigners: lambda = {lam}")

        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)

        wig = open(osp.join(
            inp.salted.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
        ), "a")
        get_wigner3j(llmax,llvec,lam,wig)
        wig.close()

        if inp.salted.saltedtype=="density-response" and lam>0 and lam<lmax_max:

            llmax, llvec = sph_utils.get_angular_indexes_antisymmetric(lam,nang1,nang2)

            wig = open(osp.join(
                inp.salted.saltedpath, "wigners", f"wigner_antisymm_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
            ), "a")
            get_wigner3j(llmax,llvec,lam,wig)
            wig.close()

if __name__ == "__main__":
    build()
