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

nfact = {}
for spe in species:
    for l in range(lmax[spe]+1):
        prefac = 2.0**l*(2.0/np.pi)**0.75
        expalpha = 0.25*float(2*l + 3)
        for n in range(nmax[(spe,l)]):
            projector[(spe,l)][n] *= prefac*alphas[(spe,l,n)]**expalpha
        nfact[(spe,l)] = np.zeros(ncut[(spe,l)])
        for n in range(ncut[(spe,l)]):
            nfact_temp = 0.0
            for ipgf1 in range(nmax[(spe,l)]):
                for ipgf2 in range(nmax[(spe,l)]):
                    nfact_temp += projector[(spe,l)][ipgf1,n] * projector[(spe,l)][ipgf2,n] * 0.5 * special.gamma(l+1.5) / ( (alphas[(spe,l,ipgf1)] + alphas[(spe,l,ipgf2)])**(l+1.5) )
            nfact[(spe,l)][n] = np.sqrt(nfact_temp)

nRI = 0
npgf = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
        for n in range(ncut[(spe,l)]):
            nRI += 2*l+1
        npgf += nmax[(spe,l)]*(2*l+1)

coefficients = np.loadtxt(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.coeffile)
coefs_primitive = np.zeros(npgf)
iaux = 0 
iaux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = ncut[(spe,l)]*(2*l+1)
         blocksize_proj = nmax[(spe,l)]*(2*l+1)
         # contract projections
         coef_slice = np.einsum('a,ab->ab',1.0/nfact[(spe,l)],coefficients[iaux:iaux+blocksize].reshape(ncut[(spe,l)],2*l+1))
         coefs_primitive[iaux_proj:iaux_proj+blocksize_proj] = np.dot(projector[(spe,l)],coef_slice).reshape(blocksize_proj)
         iaux += blocksize
         iaux_proj += blocksize_proj

iaux = 0 
iaux_proj = 0
coefficients = np.zeros(nRI)
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
         blocksize = nmax[(spe,l)]*(2*l+1)
         blocksize_proj = ncut[(spe,l)]*(2*l+1)
         # contract projections
         coef_slice = np.einsum('a,ab->ab',normfact[(spe,l)],coefs_primitive[iaux:iaux+blocksize].reshape(nmax[(spe,l)],2*l+1))
         coefficients[iaux_proj:iaux_proj+blocksize_proj] = np.dot(projector[(spe,l)].T,coef_slice).reshape(blocksize_proj)
         iaux += blocksize
         iaux_proj += blocksize_proj

dirpath = os.path.join(inp.path2qm, inp.coefdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
np.save(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy",coefficients)


ntot = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            ntot += 2*l+1

projector = {}
ncut = {}
for spe in species:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]

# undo contraction and get coefs
coefs = np.zeros(ntot)
iaux = 0
iaux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = nmax[(spe,l)]*(2*l+1)
         blocksize_proj = ncut[(spe,l)]*(2*l+1)
         # contract projections
         coef_slice = coefficients[iaux_proj:iaux_proj+blocksize_proj].reshape(ncut[(spe,l)],2*l+1)
         coefs[iaux:iaux+blocksize] = np.dot(projector[(spe,l)],coef_slice).reshape(blocksize)
         iaux += blocksize
         iaux_proj += blocksize_proj

atomic_integrals = {}
for spe in species:
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
            # compute integral of primitive radial functions
            int_radial = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
            int_radial /= np.sqrt(inner) # normalize radial function
            atomic_integrals[(spe,l,n)] = int_radial * np.sqrt(4.0*np.pi)

total_charge = 0.0
iaux = 0
for icen in range(natoms):
    specen = symbols[icen]
    for laux in range(lmax[specen]+1):
        for naux in range(nmax[(specen,laux)]):
            if laux==0:
                total_charge += coefs[iaux] * atomic_integrals[(specen,laux,naux)]
            iaux += 2*laux+1
print("number of electrons =", total_charge)


#overlap = np.zeros((nRI, nRI)).astype(np.double)
#for i in range(nRI):
#    offset = 4 + i*((nRI+1)*8)
#    overlap[:, i] = np.fromfile(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.ovlpfile, dtype=np.float64, offset = offset, count=nRI)
#dirpath = os.path.join(inp.path2qm, "overlaps")
#if not os.path.exists(dirpath):
#    os.mkdir(dirpath)
#np.save(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy",overlap)
#
#projections = np.dot(overlap,coefficients)
#dirpath = os.path.join(inp.path2qm, inp.projdir)
#if not os.path.exists(dirpath):
#    os.mkdir(dirpath)
#np.save(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy",projections)
