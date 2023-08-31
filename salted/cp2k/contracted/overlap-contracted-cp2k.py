import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
import copy
import argparse
import ctypes
import time

from lib import ovlp2c
from lib import ovlp2cXYperiodic
from lib import ovlp2cnonperiodic

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

bohr2angs = 0.529177210670

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
[lmax,nmax] = basis.basiset(inp.dfbasis)

def cartesian_to_spherical_transformation(l):
        """Compute Cartesian to spherical transformation matrices sorting the spherical components as {-l,...,l} 
        while sorting the Cartesian components as shown in the corresponding arrays."""

        if l==0:
            # 1 Cartesian triplet
            cart_idx = [[0,0,0]]
        elif l==1:
            # 3 Cartesian triplets
            cart_idx = [[1,0,0],[0,1,0],[0,0,1]]
        elif l==2:
            # 6 Cartesian triplets
            cart_idx = [[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]]
        elif l==3:
            # 10 Cartesian triplets
            cart_idx = [[3,0,0],[2,1,0],[2,0,1],[1,2,0],[1,1,1],[1,0,2],[0,3,0],[0,2,1],[0,1,2],[0,0,3]]
        elif l==4:
            # 15 Cartesian triplets
            cart_idx = [[4,0,0],[3,1,0],[3,0,1],[2,2,0],[2,1,1],[2,0,2],[1,3,0],[1,2,1],[1,1,2],[1,0,3],[0,4,0],[0,3,1],[0,2,2],[0,1,3],[0,0,4]]
        elif l==5:
            # 21 Cartesian triplets
            cart_idx = [[0,0,5],[2,0,3],[0,2,3],[4,0,1],[0,4,1],[2,2,1],[1,0,4],[0,1,4],[3,0,2],[0,3,2],[1,2,2],[2,1,2],[5,0,0],[0,5,0],[1,4,0],[4,1,0],[3,2,0],[2,3,0],[1,1,3],[3,1,1],[1,3,1]]
        elif l==6:
            # 28 Cartesian triplets
            cart_idx = [[6,0,0],[0,6,0],[0,0,6],[5,0,1],[5,1,0],[0,5,1],[1,5,0],[0,1,5],[1,0,5],[4,0,2],[4,2,0],[0,4,2],[2,4,0],[0,2,4],[2,0,4],[4,1,1],[1,4,1],[1,1,4],[3,1,2],[1,3,2],[1,2,3],[3,2,1],[2,3,1],[2,1,3],[3,3,0],[3,0,3],[0,3,3],[2,2,2]]
        else:
            print("ERROR: Cartesian to spherical transformation not available for l=",l)

        mat = np.zeros((2*l+1,int((l+2)*(l+1)/2)),complex)
        # this implementation follows Eq.15 of SCHLEGEL and FRISH, Inter. J. Quant. Chem., Vol. 54, 83-87 (1995)
        for m in range(l+1):
            itriplet = 0
            for triplet in cart_idx:
                lx = triplet[0]
                ly = triplet[1]
                lz = triplet[2]
                sfact = np.sqrt(math.factorial(l)*math.factorial(2*lx)*math.factorial(2*ly)*math.factorial(2*lz)*math.factorial(l-m)/(math.factorial(lx)*math.factorial(ly)*math.factorial(lz)*math.factorial(2*l)*math.factorial(l+m))) / (math.factorial(l)*2**l)
                j = (lx+ly-m)/2
                if j.is_integer()==True:
                    j = int(j)
                    if j>=0:
                       for ii in range(math.floor((l-m)/2)+1):
                           if j<=ii:
                               afact = special.binom(l,ii)*special.binom(ii,j)*math.factorial(2*l-2*ii)/math.factorial(l-m-2*ii)*(-1.0)**ii
                               for k in range(j+1):
                                   kk = lx-2*k
                                   if m>=kk and kk>=0:
                                      jj = (m-kk)/2
                                      bfact = special.binom(j,k)*special.binom(m,kk)*(-1.0)**(jj)
                                      mat[l+m,itriplet] += afact*bfact
                mat[l+m,itriplet] *= sfact
                mat[l-m,itriplet] = np.conj(mat[l+m,itriplet])
                if m%2!=0:
                    mat[l+m,itriplet] *= -1.0 # TODO convention to be understood
                itriplet += 1

        return[np.asarray(mat), cart_idx]

def complex_to_real_transformation(sizes):
    """Transformation matrix from complex to real spherical harmonics"""
    matrices = []
    for i in range(len(sizes)):
        lval = int((sizes[i]-1)/2)
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range(lval):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[lval][lval] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)
    return matrices

# init geometry
geom = xyzfile[iconf]
geom.wrap()
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
    avals = np.loadtxt("alphas-"+str(spe)+".txt")
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            alphas[(spe,l,n)] = avals[n]
            sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr
            rcuts[(spe,l,n)] = np.sqrt(float(2+l)/(2*alphas[(spe,l,n)]))*4
    #with open("BASIS_LRIGPW_AUXMOLOPT") as f:
    #     for line in f:
    #         if line.rstrip().split()[0] == spe and line.rstrip().split()[-1] == inp.dfbasis:
    #            nalphas = int(list(islice(f, 1))[0])
    #            lines = list(islice(f, 1+2*nalphas))
    #            nval = {}
    #            for l in range(lmax[spe]+1):
    #                nval[l] = 0
    #            for ialpha in range(nalphas):
    #                alpha = np.array(lines[1+2*ialpha].split())[0]
    #                lbools = np.array(lines[1+2*ialpha].split())[1:]
    #                l = 0
    #                for ibool in lbools:
    #                    alphas[(spe,l,nval[l])] = float(alpha)
    #                    sigmas[(spe,l,nval[l])] = np.sqrt(0.5/alphas[(spe,l,nval[l])]) # bohr
    #                    rcuts[(spe,l,nval[l])] = np.sqrt(float(2+l)/(2*alphas[(spe,l,nval[l])]))*4
    #                    nval[l]+=1
    #                    l += 1
    #            break
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

llist = []
for spe in species:
    llist.append(lmax[spe])
llmax = max(llist)

renorm = {}
for l in range(llmax+1):
    triplets = cartesian_to_spherical_transformation(l)[1]
    renorm[l] = np.zeros((2*l+1,int((l+1)*(l+2)/2)))
    itriplet = 0
    for triplet in triplets:
        lx = triplet[0]
        ly = triplet[1]
        lz = triplet[2]
        renormfact = math.factorial(2*l+2) * math.factorial(lx) * math.factorial(ly) * math.factorial(lz)
        renormfact /= 8*np.pi * math.factorial(l+1) * math.factorial(2*lx) * math.factorial(2*ly) * math.factorial(2*lz)
        renorm[l][:,itriplet] = np.sqrt(renormfact)
        itriplet += 1

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

contr_over = np.zeros((naux_proj,naux_proj))
iaux_proj = 0
for iat in range(natoms):
    print(iat+1,flush=True)
    spe1 = symbols[iat]
    coord1 = coords[iat]
    for l1 in range(lmax[spe1]+1):
        ncart1 = int((2+l1)*(1+l1)/2)
        cartidx1 = np.array(cartesian_to_spherical_transformation(l1)[1])
        c2r = complex_to_real_transformation([2*l1+1])[0]
        c2s_complex = cartesian_to_spherical_transformation(l1)[0]
        c2s_real1 = np.multiply(np.real(np.dot(c2r,c2s_complex)),renorm[l1])
        blocksize = nmax[(spe1,l1)]*(2*l1+1)
        blocksize_proj = ncut[(spe1,l1)]*(2*l1+1)
        over = np.zeros((blocksize,ntot))
        iaux1 = 0
        for n1 in range(nmax[(spe1,l1)]):
            inner1 = 0.5*special.gamma(l1+1.5)*(sigmas[(spe1,l1,n1)]**2)**(l1+1.5)
            iaux2 = 0
            for jat in range(natoms):
                spe2 = symbols[jat]
                coord2 = coords[jat]
                for l2 in range(lmax[spe2]+1):
                    cartidx2 = np.array(cartesian_to_spherical_transformation(l2)[1])
                    ncart2 = int((2+l2)*(1+l2)/2)
                    c2r = complex_to_real_transformation([2*l2+1])[0]
                    c2s_complex = cartesian_to_spherical_transformation(l2)[0]
                    c2s_real2 = np.multiply(np.real(np.dot(c2r,c2s_complex)),renorm[l2])
                    for n2 in range(nmax[(spe2,l2)]):
                        inner2 = 0.5*special.gamma(l2+1.5)*(sigmas[(spe2,l2,n2)]**2)**(l2+1.5)
                        repmax = np.zeros(3,int)
                        for ix in range(3):
                            nreps = math.ceil((rcuts[(spe1,l1,n1)]+rcuts[(spe2,l2,n2)])/cell[ix,ix])
                            if nreps < 1:
                                repmax[ix] = 1
                            else:
                                repmax[ix] = nreps
                        if inp.periodic=="3D":
                            ovlp_2c_cart = ovlp2c.ovlp2c(ncart1,ncart2,coord1,coord2,cell,repmax,alphas[(spe1,l1,n1)],alphas[(spe2,l2,n2)],cartidx1.T,cartidx2.T)
                        elif inp.periodic=="2D":
                            ovlp_2c_cart = ovlp2cXYperiodic.ovlp2c(ncart1,ncart2,coord1,coord2,cell,repmax,alphas[(spe1,l1,n1)],alphas[(spe2,l2,n2)],cartidx1.T,cartidx2.T)
                        elif inp.periodic=="0D":
                            ovlp_2c_cart = ovlp2cnonperiodic.ovlp2c(ncart1,ncart2,coord1,coord2,cell,repmax,alphas[(spe1,l1,n1)],alphas[(spe2,l2,n2)],cartidx1.T,cartidx2.T)
                        else:
                            print("ERROR: selected periodicity not implemented.")
                            sys.exit(0)
                        ovlp_2c_cart = np.transpose(ovlp_2c_cart,(1,0))
                        # convert to spherical auxiliary functions
                        ovlp_2c = np.dot(c2s_real1,np.dot(ovlp_2c_cart,c2s_real2.T))
                        # normalize auxiliary functions
                        ovlp_2c /= np.sqrt(inner1*inner2)
                        # compute density projections 
                        over[iaux1:iaux1+2*l1+1,iaux2:iaux2+2*l2+1] = ovlp_2c
                        iaux2 += 2*l2+1
            iaux1 += 2*l1+1
        ovlp_slice = over.reshape(nmax[(spe1,l1)],2*l1+1,over.shape[-1])
        contr_over_temp = np.einsum('ab,bmo->amo',projector[(spe1,l1)].T,ovlp_slice).reshape(blocksize_proj,over.shape[-1])
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
        iaux_proj += blocksize_proj

# save overlap matrix
dirpath = os.path.join(inp.path2qm, "overlaps")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
np.save(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy",contr_over)
