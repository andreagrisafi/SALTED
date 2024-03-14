import argparse
import os
import sys
import glob
import re
from multiprocessing import Pool, cpu_count
import tqdm
import numpy as np
from ase.io import read
from pyscf import gto
from pyscf import scf,dft
from pyscf import lib
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
dirpath = os.path.join(inp.path2qm, "density_matrices")

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

if iconf != -1:
    print("Calculating density matrix for configuration", iconf)
    iconf -= 1 # 0-based indexing
    conf_list = [iconf]
else:
    conf_list = range(len(geoms))

#See if any structures already exist, if they do, do not compute again.
alreadyCalculated = np.array([re.findall(r'\d+',s) for s in glob.glob(f"{dirpath}/*")], dtype=int).flatten()-1
if len(alreadyCalculated) > 0:
    print("Found existing calculations, resuming from bevore")
    conf_list = list(conf_list)
    for i in alreadyCalculated:
        conf_list.remove(i)

def doSCF(i, preliminary = False):
    geom = geoms[i]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for j in range(natoms):
        coord = coords[j]
        atoms.append([symb[j],(coord[0],coord[1],coord[2])])

    # Get PySCF objects for wave-function and density-fitted basis
    mol = gto.M(atom=atoms,basis=inp.qmbasis)
    mol.verbose = 0
    m = dft.rks.RKS(mol)
    
    if "r2scan" in inp.functional.lower():
        m._numint.libxc = dft.xcfun
    m.grids.radi_method = dft.gauss_chebyshev
    m.grids.level = 0
    m = m.density_fit()
    m.with_df.auxbasis = 'def2-tzvp-jkfit'
    m.xc = inp.functional
    #Read checkpoint from a preliminary run
    m.chkfile = 'start_checkpoint'
    m.init_guess = "chkfile"
    if i % 100 == 0:
        m.kernel()
    else:
        m.kernel(dump_chk = False)

    dm = m.make_rdm1()
    np.save(os.path.join(dirpath, f"dm_conf{i+1}.npy"), dm)



lib.num_threads(22)

for i in tqdm.tqdm(conf_list,total=len(conf_list)):
    doSCF(i)

