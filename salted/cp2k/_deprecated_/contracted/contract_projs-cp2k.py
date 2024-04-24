import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
from scipy.optimize import minimize
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
    return iconf

args = add_command_line_arguments("")
iconf = set_variable_values(args)

#print("conf", iconf)
iconf -= 1 # 0-based indexing 

bohr2angs = 0.529177210670

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
[lmax,nmax] = basis.basiset(inp.dfbasis)

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
print("Reading auxiliary basis info...")
alphas = {}
sigmas = {}
for spe in species:
    avals = np.loadtxt("alphas-"+str(spe)+".txt")
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            alphas[(spe,l,n)] = avals[n]
            sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

# init geometry
geom = xyzfile[iconf]
geom.wrap()
symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

aux_projs = np.load(inp.path2qm+inp.primprojdir+"projections_conf"+str(iconf)+".npy")
print("starting dimension:",len(aux_projs))

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
print("final dimension:",naux_proj,flush=True)
    
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
         proj_slice = aux_projs[iaux:iaux+blocksize].reshape(nmax[(spe,l)],2*l+1)
         contr_proj[iaux_proj:iaux_proj+blocksize_proj] = np.dot(projector[(spe,l)].T,proj_slice).reshape(blocksize_proj)
         iaux += blocksize
         iaux_proj += blocksize_proj
    
dirpath = os.path.join(inp.path2qm, inp.projdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath) 
np.save(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy",contr_proj)
