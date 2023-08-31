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

import pathlib
SALTEDPATHLIB = str(pathlib.Path(__file__).parent.resolve())+"/../../"
sys.path.append(SALTEDPATHLIB)
from lib import ovlp2c
from lib import ovlp2cXYperiodic
from lib import ovlp2cnonperiodic

SALTEDPATHLIB = str(pathlib.Path(__file__).parent.resolve())+"/../"
sys.path.append(SALTEDPATHLIB)
import basis
import sph_utils

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
    for l in range(lmax[spe]+1):
        avals = np.loadtxt(spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat")
        if nmax[(spe,l)]==1:
            alphas[(spe,l,0)] = float(avals)
            sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
            rcuts[(spe,l,0)] = np.sqrt(float(2+l)/(2*alphas[(spe,l,0)]))*4
        else:
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
    triplets = sph_utils.cartesian_to_spherical_transformation(l)[1]
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

over = np.zeros((ntot,ntot))
iaux1 = 0
for iat in range(natoms):
    print(iat+1,flush=True)
    spe1 = symbols[iat]
    coord1 = coords[iat]
    for l1 in range(lmax[spe1]+1):
        ncart1 = int((2+l1)*(1+l1)/2)
        cartidx1 = np.array(sph_utils.cartesian_to_spherical_transformation(l1)[1])
        c2r = sph_utils.complex_to_real_transformation([2*l1+1])[0]
        c2s_complex = sph_utils.cartesian_to_spherical_transformation(l1)[0]
        c2s_real1 = np.multiply(np.real(np.dot(c2r,c2s_complex)),renorm[l1])
        for n1 in range(nmax[(spe1,l1)]):
            inner1 = 0.5*special.gamma(l1+1.5)*(sigmas[(spe1,l1,n1)]**2)**(l1+1.5)
            iaux2 = 0
            for jat in range(natoms):
                spe2 = symbols[jat]
                coord2 = coords[jat]
                for l2 in range(lmax[spe2]+1):
                    cartidx2 = np.array(sph_utils.cartesian_to_spherical_transformation(l2)[1])
                    ncart2 = int((2+l2)*(1+l2)/2)
                    c2r = sph_utils.complex_to_real_transformation([2*l2+1])[0]
                    c2s_complex = sph_utils.cartesian_to_spherical_transformation(l2)[0]
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

# save overlap matrix
dirpath = os.path.join(inp.saltedpath, "overlaps")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
np.save(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy",over)
