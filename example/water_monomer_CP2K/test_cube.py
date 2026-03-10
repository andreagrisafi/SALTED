import sys
import numpy as np
from ase.io import read
from salted.cp2k import df2cube
from salted.sys_utils import ParseConfig

inp = ParseConfig().parse_input()

if inp.system.parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    comm = None
    size = 1
    rank = 0

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
