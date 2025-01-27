import sys
import numpy as np
from ase.io import read
from salted.cp2k import df2cube
from salted.sys_utils import ParseConfig

inp = ParseConfig().parse_input()

# Load structure
iconf = 0 
structure = read(inp.system.filename,":")[iconf]

# Cube file name
cubename = "ghost-density_conf"+str(iconf)+".cube"

# Provide reference cube filename if existing
refcube = []

# Load coefficients
coefs = np.load("coefficients_conf0.npy")

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

df2cube.build(structure,coefs,cubename,refcube,comm,size,rank)
