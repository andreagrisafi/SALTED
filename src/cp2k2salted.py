import os
import sys
import numpy as np
from pyscf import gto
from ase.io import read
from scipy import special
import argparse

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

print "conf", iconf
iconf -= 1 # 0-based indexing 

import basis

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
# define CP2K dictionary of atomic species #TODO automatize this
ispe = {}
ispe["O"] = 1
ispe["H"] = 2

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

# get maximum values for lmax and nmax
llist = []
nlist = []
for spe in species:
    llist.append(lmax[spe])
    for l in xrange(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# init geometry
geom = xyzfile[iconf]
symbols = geom.get_chemical_symbols()
coords = geom.get_positions()
natoms = len(coords)

# get CP2K information for accessing the LRI basis
lvals = {}
atom_count = {}
nbas = {}
for spe in species:
    f = open(spe+"-LRI-DZVP-MOLOPT-GTH-MEDIUM_info.txt","r")
    lines = f.readlines()
    f.close()
    nbas[spe] = int(lines[0]) # basis set size
    lvals[spe] = np.asarray(lines[2:],int) # list of angular momenta
    atom_count[spe] = 0

# compute total size of the coefficients vector and overlap matrix 
total_size = 0
for iatom in range(natoms):
    spe = symbols[iatom]
    total_size += nbas[spe] 

# extract coefficients and overlap from CP2K calculation
Coef = np.zeros(total_size)
Ovlp = np.zeros((total_size,total_size))
isgf = 0 # index for spherical Gaussian functions
iblock1 = 0
# loop over 1st atom
for iatom in range(natoms):
    spe1 = symbols[iatom]
    # load CP2K coefficients counting the number of atoms per species
    atom_count[spe1] += 1 
    c = np.loadtxt(inp.path2data+"runs/conf_"+str(iconf+1)+"/coefficients/coefs_type"+str(ispe[spe1])+"_atom"+str(atom_count[spe1])+".dat")
    # single out the coefficients dimensions in a multi-dimensional array
    n_count = {}
    for l in xrange(lmax[spe1]+1):
        n_count[l] = 0
    i = 0
    coeffs = np.zeros((llmax+1,nnmax,2*llmax+1))
    for l in lvals[spe1]:
        coeffs[l,n_count[l],:2*l+1] = c[i:i+2*l+1]
        n_count[l] += 1
        i += 2*l+1
    # fill coefficient vector
    for l in xrange(lmax[spe1]+1):
        for n in xrange(nmax[(spe1,l)]):
            for im in xrange(2*l+1):
                Coef[isgf] = coeffs[l,n,im]
                isgf += 1
    iblock2 = 0
    # loop over 2nd atom
    for jatom in xrange(natoms):
        spe2 = symbols[jatom]
        # load CP2K overlap atomic blocks if available
        try:
            o = np.loadtxt(inp.path2data+"runs/conf_"+str(iconf+1)+"/overlaps/overlap_"+str(iatom+1)+"-"+str(jatom+1)+".dat")
            o = o.reshape(nbas[spe1],nbas[spe2])
            transposed = False
        except:
            transposed = True
        # otherwise load transposed
        if transposed == True: 
            o = np.loadtxt(inp.path2data+"runs/conf_"+str(iconf+1)+"/overlaps/overlap_"+str(jatom+1)+"-"+str(iatom+1)+".dat")
            o = o.reshape(nbas[spe2],nbas[spe1]).T
        # single out the overlap dimensions in a multi-dimensional array
        overlap = np.zeros((llmax+1,llmax+1,nnmax,nnmax,2*llmax+1,2*llmax+1))
        n_count1 = {}
        for l1 in xrange(lmax[spe1]+1):
            n_count1[l1] = 0
        i1 = 0
        for l1 in lvals[spe1]:
            n_count2 = {}
            for l2 in xrange(lmax[spe2]+1):
                n_count2[l2] = 0
            i2 = 0
            for l2 in lvals[spe2]:
                overlap[l1,l2,n_count1[l1],n_count2[l2],:2*l1+1,:2*l2+1] = o[i1:i1+2*l1+1][:,i2:i2+2*l2+1] 
                n_count2[l2] += 1
                i2 += 2*l2+1
            n_count1[l1] += 1
            i1 += 2*l1+1
        # fill overlap matrix 
        isgf1 = iblock1 
        for l1 in xrange(lmax[spe1]+1):
            for n1 in xrange(nmax[(spe1,l1)]):
                for im1 in xrange(2*l1+1):
                    isgf2 = iblock2
                    for l2 in xrange(lmax[spe2]+1):
                        for n2 in xrange(nmax[(spe2,l2)]):
                            for im2 in xrange(2*l2+1):
                                Ovlp[isgf1,isgf2] = overlap[l1,l2,n1,n2,im1,im2]
                                isgf2 += 1
                    isgf1 += 1
        iblock2 += nbas[spe2] 
    iblock1 += nbas[spe1] 

# Compute density projections on auxiliary functions
Proj = np.dot(Ovlp,Coef)

# Make data directories if not already existing
dirpath = os.path.join(inp.path2data, "projections")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2data, "overlaps")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# Save projections and overlaps
np.save(inp.path2data+"projections/projections_conf"+str(iconf)+".npy",Proj)
np.save(inp.path2data+"overlaps/overlap_conf"+str(iconf)+".npy",Ovlp)
