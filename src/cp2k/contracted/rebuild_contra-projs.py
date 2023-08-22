import os
import sys
import math
import numpy as np
from ase.io import read
import copy
import argparse
import time

import basis

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return [iconf] 

args = add_command_line_arguments("")
[iconf] = set_variable_values(args)

print("conf", iconf)
iconf -= 1 # 0-based indexing 

sys.path.insert(0, './')
import inp

species = inp.species

qmpath = inp.path2qm

xyzfile = read(inp.filename,":")
[lmax,nmax] = basis.basiset(inp.dfbasis)

# init geometry
geom = xyzfile[iconf]
symbols = geom.get_chemical_symbols()
natoms = len(symbols)

# compute total number of auxiliary functions 
psize = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            psize += 2*l+1


dirpath = os.path.join(qmpath, inp.projdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

projs = np.zeros(psize)
i = 0
for iat in range(natoms):
    projs_perat = np.load(qmpath+inp.projdir+"projections_conf"+str(iconf)+"_atom"+str(iat)+".npy")
    psize_perat = len(projs_perat)
    projs[i:i+psize_perat] = projs_perat
    i += psize_perat 

projector = {}
ncut = {}
for spe in species:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]

naux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         naux_proj += ncut[(spe,l)]*(2*l+1)

# project overlap over most relevant radial channels
contr_proj = np.zeros(naux_proj)
iaux = 0
iaux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = nmax[(spe,l)]*(2*l+1)
         blocksize_proj = ncut[(spe,l)]*(2*l+1)
         # contract projections
         proj_slice = projs[iaux:iaux+blocksize].reshape(nmax[(spe,l)],2*l+1)
         contr_proj[iaux_proj:iaux_proj+blocksize_proj] = np.dot(projector[(spe,l)].T,proj_slice).reshape(blocksize_proj)
         iaux += blocksize
         iaux_proj += blocksize_proj

np.save(qmpath+inp.projdir+"projections_conf"+str(iconf)+".npy",contr_proj)
