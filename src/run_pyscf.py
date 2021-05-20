import os
import sys
import numpy as np
from pyscf import gto
from pyscf import scf,dft
from ase.io import read
from scipy import special
import argparse


def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf 

args = add_command_line_arguments("")
iconf = set_variable_values(args)

print "conf", iconf
iconf -= 1 # 0-based indexing

sys.path.insert(0, './')
import inp
# Initialize geometry
geoms = read(inp.filename,":")
geom = geoms[iconf]
symb = geom.get_chemical_symbols()
coords = geom.get_positions()
natoms = len(coords)
atoms = []
for i in range(natoms):
    coord = coords[i]
    atoms.append([symb[i],(coord[0],coord[1],coord[2])])

# Get PySCF objects for wave-function and density-fitted basis
mol = gto.M(atom=atoms,basis=inp.qmbasis)
# Run restricted hartree-fock calculation
#m = scf.RHF(mol)
m = dft.RKS(mol)
m.xc = inp.functional 
# Save density matrix
m.kernel()
dm = m.make_rdm1()

dirpath = os.path.join(inp.path2indata, "density_matrices")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

np.save(inp.path2indata+"density_matrices/dm_conf"+str(iconf+1)+".npy",dm)
