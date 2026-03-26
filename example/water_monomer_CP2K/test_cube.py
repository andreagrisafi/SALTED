import sys
import numpy as np
from ase.io import read
from salted.cp2k import df2cube

from salted.sys_utils import ParseConfig, detect_mpi
inp = ParseConfig().parse_input()

comm, size, rank, _ = detect_mpi()

# Load structure
iconf = 1 
structure = read(inp.system.filename,":")[iconf]

# Cube file name
cubename = "density_conf"+str(iconf)+".cube"

# Provide reference cube filename if existing
refcube = []

# Load coefficients
coefs = np.load("coefficients_conf"+str(iconf)+".npy")


df2cube.build(structure,coefs,cubename,refcube,comm,size,rank)
