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

SALTEDPATHLIB = str(pathlib.Path(__file__).parent.resolve())+"/../"
sys.path.append(SALTEDPATHLIB)
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
alphas = {}
sigmas = {}
for spe in species:
    for l in range(lmax[spe]+1):
        avals = np.loadtxt(spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat")
        if nmax[(spe,l)]==1:
            alphas[(spe,l,0)] = float(avals)
            sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
        else:
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

projector = {}
ncut = {}
normfact = {}
for spe in species:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]
        normfact[(spe,l)] = np.zeros(nmax[(spe,l)])
        for n in range(nmax[(spe,l)]):
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
            normfact[(spe,l)][n] = np.sqrt(inner)

norm_projector = {}
norm_primitive = {}
nfact = {}
for spe in species:
    for l in range(lmax[spe]+1):
        prefac = 2.0**l*(2.0/np.pi)**0.75
        expalpha = 0.25*float(2*l + 3)
        norm_projector[(spe,l)] = np.zeros((nmax[(spe,l)],ncut[(spe,l)]))
        norm_primitive[(spe,l)] = np.zeros(nmax[(spe,l)])
        for n in range(nmax[(spe,l)]):
            norm_primitive[(spe,l)][n] = 1.0/(prefac*alphas[(spe,l,n)]**expalpha)
            norm_projector[(spe,l)][n] = projector[(spe,l)][n] * prefac*alphas[(spe,l,n)]**expalpha
        nfact[(spe,l)] = np.zeros(ncut[(spe,l)])
        for n in range(ncut[(spe,l)]):
            nfact_temp = 0.0
            for ipgf1 in range(nmax[(spe,l)]):
                for ipgf2 in range(nmax[(spe,l)]):
                    nfact_temp += norm_projector[(spe,l)][ipgf1,n] * norm_projector[(spe,l)][ipgf2,n] * 0.5 * special.gamma(l+1.5) / ( (alphas[(spe,l,ipgf1)] + alphas[(spe,l,ipgf2)])**(l+1.5) )
            nfact[(spe,l)][n] = np.sqrt(nfact_temp)

aux_projs = np.loadtxt(inp.path2qm+"conf_"+str(iconf+1)+"/Au-RI_DENSITY_COEFFS.dat")
#aux_projs = np.loadtxt(inp.path2qm+"conf_"+str(iconf+1)+"/efield/Au-RI_DENSITY_COEFFS.dat")
print("starting dimension:",len(aux_projs))

naux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         naux_proj += ncut[(spe,l)]*(2*l+1)
print("final dimension:",naux_proj,flush=True)
    
# contract 
contr_proj = np.zeros(naux_proj)
iaux = 0
iaux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = nmax[(spe,l)]*(2*l+1)
         blocksize_proj = ncut[(spe,l)]*(2*l+1)
         proj_slice = aux_projs[iaux:iaux+blocksize].reshape(nmax[(spe,l)],2*l+1)
         contr_proj[iaux_proj:iaux_proj+blocksize_proj] = np.dot(projector[(spe,l)].T,proj_slice).reshape(blocksize_proj)
         iaux += blocksize
         iaux_proj += blocksize_proj

# rescale for different CP2K normalization
iaux = 0
contra_coefs_renorm = np.zeros(naux_proj)
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = ncut[(spe,l)]*(2*l+1)
         coefs_primit = np.einsum('a,ab->ab',norm_primitive[(spe,l)]/normfact[(spe,l)],np.dot(projector[(spe,l)],contr_proj[iaux:iaux+blocksize].reshape(ncut[(spe,l)],2*l+1)))
         contra_coefs_renorm[iaux:iaux+blocksize]  = np.einsum('a,ab->ab',nfact[(spe,l)],np.dot(projector[(spe,l)].T,coefs_primit)).reshape(blocksize)
         iaux += blocksize

np.savetxt(inp.path2qm+"conf_"+str(iconf+1)+"/Au-RI_DENSITY_COEFFS_CONTRACTED.dat",contra_coefs_renorm)
#np.savetxt(inp.path2qm+"conf_"+str(iconf+1)+"/efield/Au-RI_DENSITY_COEFFS_CONTRACTED.dat",contra_coefs_renorm)
