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
    for l in range(lmax[spe]+1):
        avals = np.loadtxt(spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat")
        if nmax[(spe,l)]==1:
            alphas[(spe,l,0)] = float(avals)
            sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
        else:
            for n in range(nmax[(spe,l)]):
                alphas[(spe,l,n)] = avals[n]
                sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

cova = {}
for spe in species:
    for l in range(lmax[spe]+1):
        cova[(spe,l)] = np.zeros((nmax[(spe,l)],nmax[(spe,l)])) 

for iconf in range(ndata):
    # init geometry
    geom = xyzfile[iconf]
    geom.wrap()
    symbols = geom.get_chemical_symbols()
    valences = geom.get_atomic_numbers()
    coords = geom.get_positions()/bohr2angs
    cell = geom.get_cell()/bohr2angs
    natoms = len(coords)

    coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    aux_projs = np.load(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy")

    iaux = 0
    for iat in range(natoms):
        spe = symbols[iat]
        for l in range(lmax[spe]+1):
            # compute inner product of contracted and normalized primitive GTOs
            rsize = nmax[(spe,l)]*(2*l+1)
            ps = aux_projs[iaux:iaux+rsize].reshape(nmax[(spe,l)],2*l+1)
            cs = coefs[iaux:iaux+rsize].reshape(nmax[(spe,l)],2*l+1)
            cova[(spe,l)] += np.dot(cs,ps.T)
            iaux += rsize

for spe in species:
    for l in range(lmax[spe]+1):
        cova[(spe,l)] /= float(ndata) 

projector = {}
ncut = {}
for spe in species:
    for l in range(lmax[spe]+1):
        print(np.allclose(cova[(spe,l)],cova[(spe,l)].T))
        eigenvalues, unitary = np.linalg.eigh(cova[(spe,l)])
        np.savetxt("eigenvalues/eigen_spe"+str(spe)+"_l"+str(l)+".txt",eigenvalues)
        if l==0:
            eigenvalues = eigenvalues[-8:]
        elif l==1:
            eigenvalues = eigenvalues[-8:]
        elif l==2:
            eigenvalues = eigenvalues[-8:]
        elif l==3:
            eigenvalues = eigenvalues[-8:]
        elif l==4:
            eigenvalues = eigenvalues[-7:]
        elif l==5:
            eigenvalues = eigenvalues[-1:]
        elif l==6:
            eigenvalues = eigenvalues[-1:]
        else:
            eigenvalues = eigenvalues[eigenvalues>inp.cutradial]
        #eigenvalues = eigenvalues[eigenvalues>inp.cutradial]
        ncut[(spe,l)] = len(eigenvalues)
        print("number of eigenvalues for ", spe,l,":",ncut[(spe,l)])
        projector[(spe,l)] = unitary[:,-ncut[(spe,l)]:] 
        np.save("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy",projector[(spe,l)])
        np.savetxt("contractions/contra_spe"+str(spe)+"_l"+str(l)+".dat",projector[(spe,l)])


for iconf in range(ndata):
    # init geometry
    geom = xyzfile[iconf]
    geom.wrap()
    symbols = geom.get_chemical_symbols()
    valences = geom.get_atomic_numbers()
    coords = geom.get_positions()/bohr2angs
    cell = geom.get_cell()/bohr2angs
    natoms = len(coords)

    coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    aux_projs = np.load(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy")
    over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    print("starting dimension:",len(aux_projs))

    naux_proj = 0
    for iat in range(natoms):
        spe = symbols[iat]
        for l in range(lmax[spe]+1):
             naux_proj += ncut[(spe,l)]*(2*l+1)
    print("final dimension:",naux_proj,flush=True)
    
    # contract projections and overlap
    contr_over = np.zeros((naux_proj,naux_proj))
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
             # contract overlap 
             ovlp_slice = over[iaux:iaux+blocksize,:].reshape(nmax[(spe,l)],2*l+1,over.shape[-1])
             contr_over_temp = np.einsum('ab,bmo->amo',projector[(spe,l)].T,ovlp_slice).reshape(blocksize_proj,over.shape[-1])
             iaux2 = 0
             iaux_proj2 = 0
             for jat in range(natoms):
                 spe2 = symbols[jat]
                 for l2 in range(lmax[spe2]+1):
                     blocksize2 = nmax[(spe2,l2)]*(2*l2+1)
                     blocksize_proj2 = ncut[(spe2,l2)]*(2*l2+1)
                     ovlp_slice2 = contr_over_temp[:,iaux2:iaux2+blocksize2].reshape(blocksize_proj,nmax[(spe2,l2)],2*l2+1)
                     contr_over[iaux_proj:iaux_proj+blocksize_proj,iaux_proj2:iaux_proj2+blocksize_proj2] = np.einsum('ab,obm->oam',projector[(spe2,l2)].T,ovlp_slice2).reshape(blocksize_proj,blocksize_proj2)
                     iaux2 += blocksize2
                     iaux_proj2 += blocksize_proj2
             iaux += blocksize
             iaux_proj += blocksize_proj
    
    #print("condition number:",np.linalg.cond(contr_over))
   
    contr_coefs = np.linalg.solve(contr_over,contr_proj)
 
    # undo contraction and get coefs
    coefs = np.zeros(len(aux_projs))
    iaux = 0
    iaux_proj = 0
    for iat in range(natoms):
        spe = symbols[iat]
        for l in range(lmax[spe]+1):
             blocksize = nmax[(spe,l)]*(2*l+1)
             blocksize_proj = ncut[(spe,l)]*(2*l+1)
             # contract projections
             coef_slice = contr_coefs[iaux_proj:iaux_proj+blocksize_proj].reshape(ncut[(spe,l)],2*l+1)
             coefs[iaux:iaux+blocksize] = np.dot(projector[(spe,l)],coef_slice).reshape(blocksize)
             iaux += blocksize
             iaux_proj += blocksize_proj
    #np.save(inp.path2qm+"coefs_contra/coefficients_conf"+str(iconf)+".npy",coefs)
    
    atomic_integrals = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            prefac = 2.0**l*(2.0/np.pi)**0.75
            expalpha = 0.25*float(2*l + 3)
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
   
