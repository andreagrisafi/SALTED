import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
from pyscf.pbc import gto
import copy
import argparse
import ctypes
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

print("conf", iconf)
iconf -= 1 # 0-based indexing 

bohr2angs = 0.529177249

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
[lmax,nmax] = basis.basiset(inp.dfbasis)

# init geometry
geom = xyzfile[iconf]
symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
print("Reading auxiliary basis info...")
alphas = {}
sigmas = {}
rcuts = {}
for spe in species:
    with open("BASIS_LRIGPW_AUXMOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[-1] == inp.dfbasis:
                nalphas = int(list(islice(f, 1))[0])
                lines = list(islice(f, 1+2*nalphas))
                nval = {}
                for l in range(lmax[spe]+1):
                    nval[l] = 0
                for ialpha in range(nalphas):
                    alpha = np.array(lines[1+2*ialpha].split())[0]
                    lbools = np.array(lines[1+2*ialpha].split())[1:]
                    l = 0
                    for ibool in lbools:
                        alphas[(spe,l,nval[l])] = float(alpha)
                        sigmas[(spe,l,nval[l])] = np.sqrt(0.5/alphas[(spe,l,nval[l])]) # bohr
                        rcuts[spe] = sigmas[(spe,l,nval[l])]*6.0 # bohr
                        nval[l]+=1
                        l += 1
                break
# compute total number of auxiliary functions 
ntot = 0
ntot_at = {}
for iat in range(natoms):
    spe = symbols[iat]
    ntot_at[iat] = 0
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            ntot += 2*l+1
            ntot_at[iat] += 2*l+1
print("number of auxiliary functions =", ntot)

print("Computing auxiliary overlap from PySCF functions...")
# define atom object for pyscf
atoms = []
for iat in range(natoms):
    coord = coords[iat] * bohr2angs
    atoms.append([symbols[iat],(coord[0],coord[1],coord[2])])

cp2kbasis = {'O': gto.basis.parse("""
O  LRI-DZVP-MOLOPT-GTH-MEDIUM
   15
 2   0   0   1  1
   24.031909411024   1.0
 2   0   0   1  1
   16.167926705922   1.0
 2   0   1   1  1  1
   10.877281929506   1.0   1.0
 2   0   2   1  1  1  1
    7.317899463920   1.0   1.0   1.0
 2   0   3   1  1  1  1  1
    4.923256831173   1.0   1.0   1.0   1.0
 2   0   3   1  1  1  1  1
    3.312215198528   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    2.228356126354   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    1.499169204968   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    1.008594756710   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    0.678551413605   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    0.456508441910   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    0.307124785767   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    0.206624073890   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    0.139010297734   1.0   1.0   1.0   1.0   1.0
 2   0   4   1  1  1  1  1  1
    0.093521836600   1.0   1.0   1.0   1.0   1.0
"""), 'H': gto.basis.parse("""
H  LRI-DZVP-MOLOPT-GTH-MEDIUM
   10
 2   0   0   1  1
   22.956000679816   1.0
 2   0   1   1  1  1
   11.437045132575   1.0   1.0
 2   0   2   1  1  1  1
    5.698118029748   1.0   1.0   1.0
 2   0   2   1  1  1  1
    2.838893149810   1.0   1.0   1.0
 2   0   3   1  1  1  1  1
    1.414381779030   1.0   1.0   1.0  1.0
 2   0   3   1  1  1  1  1
    0.704667527549   1.0   1.0   1.0  1.0
 2   0   3   1  1  1  1  1
    0.351076584656   1.0   1.0   1.0  1.0
 2   0   3   1  1  1  1  1
    0.174911945670   1.0   1.0   1.0  1.0
 2   0   3   1  1  1  1  1
    0.087143916955   1.0   1.0   1.0  1.0
 2   0   3   1  1  1  1  1
    0.043416487268   1.0   1.0   1.0  1.0
""")}

mol = gto.M(atom=atoms,basis=cp2kbasis)

# overlap of central cell 
overlap = mol.intor('int1e_ovlp_sph')

spemax = max(rcuts,key=rcuts.get)
dmax = 2*rcuts[spemax]
nreps = math.ceil(dmax/cell[0,0])
if nreps < 1:
    repmax = 1
else:
    repmax = math.ceil(dmax/cell[0,0])

print("number of cell repetitions:", repmax)

# append atom objects for periodic images
for ix in range(-repmax,repmax+1):
    for iy in range(-repmax,repmax+1):
        for iz in range(-repmax,repmax+1):
            if ix==0 and iy==0 and iz==0:
                continue
            else:
                patoms = []
                for iat in range(natoms):
                    coord = coords[iat].copy() * bohr2angs
                    patoms.append([symbols[iat],(coord[0],coord[1],coord[2])])               
                for iat in range(natoms):
                    coord = coords[iat].copy() * bohr2angs
                    coord[0] += ix*cell[0,0] * bohr2angs 
                    coord[1] += iy*cell[1,1] * bohr2angs
                    coord[2] += iz*cell[2,2] * bohr2angs
                    patoms.append([symbols[iat],(coord[0],coord[1],coord[2])])               
                pmol = gto.M(atom=patoms,basis=cp2kbasis)
                poverlap = pmol.intor('int1e_ovlp_sph')
                overlap += poverlap[:ntot,ntot:]

# reorder P-entries in overlap matrix
over = np.zeros((ntot,ntot))
i1 = 0
for iat in range(natoms):
    spe1 = symbols[iat]
    for l1 in range(lmax[spe1]+1):
        for n1 in range(nmax[(spe1,l1)]):
            for im1 in range(2*l1+1):
                i2 = 0
                for jat in range(natoms):
                    spe2 = symbols[jat]
                    for l2 in range(lmax[spe2]+1):
                        for n2 in range(nmax[(spe2,l2)]):
                            for im2 in range(2*l2+1):
                                if l1==1 and im1!=2 and l2!=1:
                                    over[i1,i2] = overlap[i1+1,i2]
                                elif l1==1 and im1==2 and l2!=1:
                                    over[i1,i2] = overlap[i1-2,i2]
                                elif l2==1 and im2!=2 and l1!=1:
                                    over[i1,i2] = overlap[i1,i2+1]
                                elif l2==1 and im2==2 and l1!=1:
                                    over[i1,i2] = overlap[i1,i2-2]
                                elif l1==1 and im1!=2 and l2==1 and im2!=2:
                                    over[i1,i2] = overlap[i1+1,i2+1]
                                elif l1==1 and im1!=2 and l2==1 and im2==2:
                                    over[i1,i2] = overlap[i1+1,i2-2]
                                elif l1==1 and im1==2 and l2==1 and im2!=2:
                                    over[i1,i2] = overlap[i1-2,i2+1]
                                elif l1==1 and im1==2 and l2==1 and im2==2:
                                    over[i1,i2] = overlap[i1-2,i2-2]
                                else:
                                    over[i1,i2] = overlap[i1,i2]
                                i2 += 1
                i1 += 1

projector = {}
ncut = {}
for spe in species:
    for l in range(lmax[spe]+1):
        # compute inner product of contracted and normalized primitive GTOs
        overl = np.zeros((nmax[(spe,l)],nmax[(spe,l)]))
        for n1 in range(nmax[(spe,l)]):
            inner1 = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n1)]**2)**(l+1.5)
            for n2 in range(nmax[(spe,l)]):
                inner2 = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n2)]**2)**(l+1.5)
                overl[n1,n2] = 0.5 * special.gamma(l+1.5) / ( (alphas[(spe,l,n1)] + alphas[(spe,l,n2)])**(l+1.5) )
                overl[n1,n2] /= np.sqrt(inner1*inner2)
        eigenvalues, unitary = np.linalg.eigh(overl)
        eigenvalues = eigenvalues[eigenvalues>inp.cutradial]
        ncut[(spe,l)] = len(eigenvalues)
        projector[(spe,l)] = unitary[:,-ncut[(spe,l)]:] 

naux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         naux_proj += ncut[(spe,l)]*(2*l+1)
 
# project overlap over most relevant radial channels
over_proj = np.zeros((naux_proj,naux_proj))
iaux = 0
iaux_proj = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
         blocksize = nmax[(spe,l)]*(2*l+1)
         blocksize_proj = ncut[(spe,l)]*(2*l+1)
         ovlp_slice = over[iaux:iaux+blocksize,:].reshape(nmax[(spe,l)],2*l+1,over.shape[-1])
         over_proj_temp = np.einsum('ab,bmo->amo',projector[spe,l].T,ovlp_slice).reshape(blocksize_proj,over.shape[-1])
         iaux2 = 0
         iaux_proj2 = 0
         for jat in range(natoms):
             spe2 = symbols[jat]
             for l2 in range(lmax[spe2]+1):
                 blocksize2 = nmax[(spe2,l2)]*(2*l2+1)
                 blocksize_proj2 = ncut[(spe2,l2)]*(2*l2+1)
                 ovlp_slice2 = over_proj_temp[:,iaux2:iaux2+blocksize2].reshape(blocksize_proj,nmax[(spe2,l2)],2*l2+1)
                 over_proj[iaux_proj:iaux_proj+blocksize_proj,iaux_proj2:iaux_proj2+blocksize_proj2] = np.einsum('ab,obm->oam',projector[spe2,l2].T,ovlp_slice2).reshape(blocksize_proj,blocksize_proj2)
                 iaux2 += blocksize2
                 iaux_proj2 += blocksize_proj2
         iaux += blocksize
         iaux_proj += blocksize_proj

print("condition number:",np.linalg.cond(over_proj))

# save overlap matrix
dirpath = os.path.join(inp.path2qm, "overlaps")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
np.save(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy",over_proj)

## define lowdin orthogonalization matrix for each angular momentum
#lowdin = {}
#for spe in species:
#    for l in range(lmax[spe]+1):
#        # compute inner product of contracted and normalized primitive GTOs
#        overl = np.zeros((nmax[(spe,l)],nmax[(spe,l)]))
#        for n1 in range(nmax[(spe,l)]):
#            inner1 = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n1)]**2)**(l+1.5)
#            for n2 in range(nmax[(spe,l)]):
#                inner2 = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n2)]**2)**(l+1.5)
#                overl[n1,n2] = 0.5 * special.gamma(l+1.5) / ( (alphas[(spe,l,n1)] + alphas[(spe,l,n2)])**(l+1.5) )
#                overl[n1,n2] /= np.sqrt(inner1*inner2)
#        eigenvalues, unitary = np.linalg.eigh(overl)
#        eigenvalues = eigenvalues[eigenvalues>1e-08]
#        Mcut = len(eigenvalues)
#        diagoverlap = np.diag(np.sqrt(1.0/eigenvalues))
#        lowdin[(spe,l)] = np.real(np.dot(np.conj(unitary[:,-Mcut:]),np.dot(diagoverlap,unitary[:,-Mcut:].T)))

##orthogonalize radial blocks along rows
#iaux = 0
#for icen in range(natoms):
#    specen = symbols[icen]
#    for laux in range(lmax[specen]+1):
#        blocksize = nmax[(specen,laux)]*(2*laux+1)
#        # select relevant slice and reshape
#        ovlp_slice = over[iaux:iaux+blocksize,:].reshape(nmax[(specen,laux)],2*laux+1,over.shape[-1])
#        # orthogonalize and reshape back
#        over[iaux:iaux+blocksize,:] = np.einsum('ab,bmo->amo',lowdin[specen,laux],ovlp_slice).reshape(blocksize,over.shape[-1])
#        iaux += blocksize
#
##orthogonalize radial blocks along cols 
#iaux = 0
#for icen in range(natoms):
#    specen = symbols[icen]
#    for laux in range(lmax[specen]+1):
#        blocksize = nmax[(specen,laux)]*(2*laux+1)
#        # select relevant slice and reshape
#        ovlp_slice = over[:,iaux:iaux+blocksize].reshape(over.shape[0],nmax[(specen,laux)],2*laux+1)
#        # orthogonalize and reshape back
#        over[:,iaux:iaux+blocksize] = np.einsum('ab,obm->oam',lowdin[(specen,laux)],ovlp_slice).reshape(over.shape[0],blocksize)
#        iaux += blocksize
#
#print("condition number:",np.linalg.cond(over))

## save overlap matrix
#dirpath = os.path.join(inp.path2qm, "overlaps")
#if not os.path.exists(dirpath):
#    os.mkdir(dirpath)
#np.save(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy",over)
