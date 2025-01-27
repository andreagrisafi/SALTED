import os
import sys
import time
import os.path as osp

import numpy as np
from ase.io import read
from scipy import special
from scipy import sparse

# Import SALTED functions
from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range, init_property_file
from salted.cp2k.utils import init_moments, init_ghost_integrals, compute_ghost_center

bohr2angs = 0.529177210670

# Get SALTED basic parameters
inp = ParseConfig().parse_input()
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

# Read cell from a reference structure
xyzfile = read(inp.system.filename, ":")
cell = xyzfile[0].get_cell()

# Initialize integrals for computing the center of charge of the ghost density
kmin_integrals,kmin_harmonics = init_ghost_integrals(inp,cell/bohr2angs,lmax,nmax,species)

# Initialize calculation of density moments (only charge integrals will be needed in this example)
charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,0)

# Possibly loop over any given structure of choice 
iconf = 0
structure = xyzfile[iconf]
atomic_symbols = structure.get_chemical_symbols()
natoms = len(atomic_symbols)

# Get coefficients for the given structure (this might be directly provided as the result of a SALTED prediction)
coefs = np.load("coefficients_conf0.npy")

# Compute center of charge of ghost density in atomic units
center_of_charge = compute_ghost_center(structure,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,kmin_integrals,kmin_harmonics,coefs)

# Convert to angstrom
center_of_charge *= bohr2angs

# Fold within cell
for ik in range(3):
    center_of_charge[ik] = center_of_charge[ik] % cell[ik,ik] 

print("Center of charge (angstrom) =", center_of_charge)
