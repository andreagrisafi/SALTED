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

from sympy.parsing import mathematica
from sympy import symbols
from sympy import lambdify


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
chemical_symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

contr_proj = np.load(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy")
contr_over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
contr_coef = np.linalg.solve(contr_over,contr_proj)
dirpath = os.path.join(inp.path2qm, inp.coefdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
np.save(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy",contr_coef)


# If total charge density is asked, read in the GTH pseudo-charge and return a radial numpy function
if inp.totcharge:
    pseudof = open(inp.pseudochargefile,"r")
    pseudochargedensity = mathematica.mathematica(pseudof.readlines()[0],{'Erf[x]':'erf(x)'})
    pseudof.close()
    rpseudo = symbols('r')
    pseudochargedensity = lambdify(rpseudo, pseudochargedensity, modules=['numpy'])
    pseudochargedensity = np.vectorize(pseudochargedensity)
    nn = 100000
    dr = 5.0/nn
    pseudocharge = 0.0
    for ir in range(1,nn):
        r = ir*dr
        pseudocharge += r**2*pseudochargedensity(r)
    pseudocharge *= 4*np.pi*dr
    print("Integrated pseudo-charge =", pseudocharge)


ntot = 0
nele = 0
for iat in range(natoms):
    spe = chemical_symbols[iat]
    nele += pseudocharge
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
    spe = chemical_symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = nmax[(spe,l)]*(2*l+1)
         blocksize_proj = ncut[(spe,l)]*(2*l+1)
         # contract projections
         coef_slice = contr_coef[iaux_proj:iaux_proj+blocksize_proj].reshape(ncut[(spe,l)],2*l+1)
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
    specen = chemical_symbols[icen]
    for laux in range(lmax[specen]+1):
        for naux in range(nmax[(specen,laux)]):
            if laux==0:
                total_charge += coefs[iaux] * atomic_integrals[(specen,laux,naux)]
            iaux += 2*laux+1
print("number of electrons =", total_charge)

# rescale for different CP2K normalization
iaux = 0
for iat in range(natoms):
    spe = chemical_symbols[iat]
    for l in range(lmax[spe]+1):
        for n in range(ncut[(spe,l)]):
            for im in range(2*l+1):
                if l==0 and im==0:
                    contr_coef[iaux] *= nele/total_charge
                iaux += 1


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


iaux = 0
coefs_renorm = np.zeros(len(contr_coef))
for iat in range(natoms):
    spe = chemical_symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = ncut[(spe,l)]*(2*l+1)
         coefs_primit = np.einsum('a,ab->ab',norm_primitive[(spe,l)]/normfact[(spe,l)],np.dot(projector[(spe,l)],contr_coef[iaux:iaux+blocksize].reshape(ncut[(spe,l)],2*l+1)))
         coefs_renorm[iaux:iaux+blocksize]  = np.einsum('a,ab->ab',nfact[(spe,l)],np.dot(projector[(spe,l)].T,coefs_primit)).reshape(blocksize)
         iaux += blocksize

# save predicted coefficients
np.savetxt(inp.path2qm+inp.coefdir+"COEFFS-"+str(iconf+1)+".dat",coefs_renorm)


