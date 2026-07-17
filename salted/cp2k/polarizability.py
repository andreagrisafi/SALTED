import os
import sys
import time
import os.path as osp

import numpy as np
from scipy import special
from scipy import sparse

from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, init_property_file
from salted.cp2k.utils import init_moments, compute_charge_and_dipole, compute_polarizability

def build(iconf,ref_coefs):
    """Compute polarizability tensor for the given structure and related set of density-response coefficients."""

    inp = ParseConfig().parse_input()

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()

    if inp.qm.qmcode=="cp2k":
        from ase.io import read
        xyzfile = read(inp.system.filename, ":")
        # Initialize calculation of density/density-response moments
        charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,0)

    ref_alpha = compute_polarizability(xyzfile[iconf],natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,ref_coefs)

    # Save polarizabilities
    return ref_alpha
    #print(ref_alpha[("x","x")],ref_alpha[("x","y")],ref_alpha[("x","z")])
    #print(ref_alpha[("y","x")],ref_alpha[("y","y")],ref_alpha[("y","z")])
    #print(ref_alpha[("z","x")],ref_alpha[("z","y")],ref_alpha[("z","z")]) 
