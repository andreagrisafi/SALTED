import os
import sys
import time
import os.path as osp

import ase
import numpy as np
from sympy.physics.wigner import wigner_3j

from salted import sph_utils
from salted import basis


def build(field):
    sys.path.insert(0, './')
    import inp

    from salted.sys_utils import read_system, get_atom_idx
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # Generate directories for saving descriptors
    dirpath = os.path.join(inp.saltedpath, "wigners")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    for lam in range(lmax_max+1):

        print(f"lambda = {lam}")

        # External field?
        if field:
            # Select relevant angular components for equivariant descriptor calculation
            llmax = 0
            lvalues = {}
            for l1 in range(inp.nang1+1):
                # keep only even combination to enforce inversion symmetry
                if (lam+l1+1)%2==0 :
                    if abs(1-lam) <= l1 and l1 <= (1+lam) :
                        lvalues[llmax] = [l1,1]
                        llmax+=1
        else:
            # Select relevant angular components for equivariant descriptor calculation
            llmax = 0
            lvalues = {}
            for l1 in range(inp.nang1+1):
                for l2 in range(inp.nang2+1):
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

        # Precompute Wigner-3J symbols and save to file as dense arrays
        if field:
            wig = open(osp.join(
                inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{inp.nang1}_field.dat"
            ), "a")
            iwig = 0
            for il in range(llmax):
                l1 = lvalues[il][0]
                l2 = lvalues[il][1]
                for imu in range(2*lam+1):
                    mu = imu-lam
                    for im1 in range(2*l1+1):
                        m1 = im1-l1
                        m2 = m1-mu
                        if abs(m2) <= l2 and m2==0:
                            im2 = m2+l2
                            w3j = wigner_3j(lam,l2,l1,mu,m2,-m1) * (-1.0)**(m1)
                            print(float(w3j),file=wig)
            wig.close()
        else:
            wig = open(osp.join(
                inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{inp.nang1}_lmax2-{inp.nang2}.dat"
            ), "a")
            iwig = 0
            for il in range(llmax):
                l1 = lvalues[il][0]
                l2 = lvalues[il][1]
                for imu in range(2*lam+1):
                    mu = imu-lam
                    for im1 in range(2*l1+1):
                        m1 = im1-l1
                        m2 = m1-mu
                        if abs(m2) <= l2:
                            im2 = m2+l2
                            w3j = wigner_3j(lam,l2,l1,mu,m2,-m1) * (-1.0)**(m1)
                            print(float(w3j),file=wig)
            wig.close()

if __name__ == "__main__":
    build()
