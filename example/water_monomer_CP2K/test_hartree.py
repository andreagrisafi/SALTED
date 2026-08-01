import os
import os.path as osp
import sys
import numpy as np
from ase.io import read
from salted.cp2k.utils import init_moments, compute_charge_and_dipole, elec_energy_forces
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, detect_mpi
from mpi4py import MPI

inp = ParseConfig().parse_input()

bohr2angs = 0.529177249

comm, size, rank, _ = detect_mpi()

# Load structure
iconf = 0
structure = read(inp.system.filename,":")[iconf]

# Initialize SALTED prediction
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

coefs = np.load("coefficients/coefficients_conf0.npy")

h, forces = elec_energy_forces(lmax,nmax,inp.salted.saltedpath,inp.qm.dfbasis,species,structure,coefs)
print("Total electrostatic energy: ", h)
