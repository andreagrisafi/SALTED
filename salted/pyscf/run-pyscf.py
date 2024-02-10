import argparse
import os
import sys

import numpy as np
from ase.io import read
from pyscf import gto
from pyscf import scf,dft
from pyscf import grad
from scipy import special

sys.path.insert(0, './')
import inp

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=-1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf 

args = add_command_line_arguments("")
iconf = set_variable_values(args)

# Initialize geometry
geoms = read(inp.filename,":")
    
dirpath = inp.path2qm
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

if iconf != -1:
    print("Calculating density matrix for configuration", iconf)
    iconf -= 1 # 0-based indexing
    conf_list = [iconf]
else:
    conf_list = range(len(geoms))

for iconf in conf_list:
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
    m = dft.RKS(mol)
    m.xc = inp.functional
    # Save density matrix
    m.kernel()
    
    #ks_scanner = m.apply(grad.RKS).as_scanner()
    #etot, grad = ks_scanner(mol)
    #
    #f = open("gradients/grad_conf"+str(iconf+1)+".dat","w")
    #for i in range(natoms):
    #    print >> f, symb[i], grad[i,0], grad[i,1], grad[i,2]
    #f.close()
    
    dm = m.make_rdm1()
    
    dirpath = os.path.join(inp.path2qm, "density_matrices")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    
    np.save(os.path.join(inp.path2qm, "density_matrices", f"dm_conf{iconf+1}.npy", dm))
