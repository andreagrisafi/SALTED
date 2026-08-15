import sys
import numpy as np
from ase.io import read
from salted.cp2k import cube_reconstruction
from numba import njit, prange
from numba import types
from numba.typed import Dict

from salted.sys_utils import ParseConfig, detect_mpi
inp = ParseConfig().parse_input()

comm, size, rank, _ = detect_mpi()

# Load structure
iconf = 0
structure = read(inp.system.filename,":")[iconf]

# Cube file name prefix
cubename = "conf"+str(iconf)+".cube"

# Provide one reference cube filename if existing
refcube = []

# Load coefficients
coefs = np.load("coefficients_conf"+str(iconf)+".npy")

f_list = ["e_density", "potential", "efield_x", "efield_y", "efield_z"]

cube_reconstruction.build(f_list,structure,coefs,cubename,refcube,comm,size,rank)
