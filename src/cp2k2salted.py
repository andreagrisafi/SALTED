import os
import sys
import numpy as np
from ase.io import read
from scipy import special
import argparse
import time
from itertools import islice

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
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# init geometry
geom = xyzfile[iconf]
symbols = geom.get_chemical_symbols()
coords = geom.get_positions()
natoms = len(coords)

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT 
lvals = {}
atom_count = {}
nbas = {}
for spe in species:
    nbas[spe] = 0
    atom_count[spe] = 0
    lvals[spe] = []
    with open("BASIS_LRIGPW_AUXMOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[-1] == "LRI-DZVP-MOLOPT-GTH-MEDIUM":
                nalphas = int(list(islice(f, 1))[0])
                lines = list(islice(f, 1+2*nalphas))
                for ialpha in range(nalphas):
                    lbools = np.array(lines[1+2*ialpha].split())[1:]
                    l = 0
                    for ibool in lbools: 
                        lvals[spe].append(l)
                        nbas[spe] += 2*l+1
                        l += 1
                break
    print(spe+" LRI-basis set size:", nbas[spe])

# compute total size of the coefficients vector and overlap matrix 
total_size = 0
for iatom in range(natoms):
    spe = symbols[iatom]
    total_size += nbas[spe] 

print("total size =", total_size)

start = time.time()

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
    c = np.loadtxt(inp.path2qm+"runs/conf_"+str(iconf+1)+"/coefficients/coefs_type"+str(ispe[spe1])+"_atom"+str(atom_count[spe1])+".dat")
    # single out the coefficients dimensions in a multi-dimensional array
    n_count = {}
    for l in range(lmax[spe1]+1):
        n_count[l] = 0
    i = 0
    coeffs = np.zeros((llmax+1,nnmax,2*llmax+1))
    for l in lvals[spe1]:
        coeffs[l,n_count[l],:2*l+1] = c[i:i+2*l+1]
        n_count[l] += 1
        i += 2*l+1
    # fill coefficient vector
    for l in range(lmax[spe1]+1):
        for n in range(nmax[(spe1,l)]):
            for im in range(2*l+1):
                Coef[isgf] = coeffs[l,n,im]
                isgf += 1
    iblock2 = 0
    # loop over 2nd atom
    for jatom in range(natoms):
        spe2 = symbols[jatom]
        # load CP2K overlap atomic blocks if available
        transposed = False
        try:
            o = np.loadtxt(inp.path2qm+"runs/conf_"+str(iconf+1)+"/overlaps/overlap_"+str(iatom+1)+"-"+str(jatom+1)+".dat")
            o = o.reshape(nbas[spe1],nbas[spe2])
        except:
            transposed = True
        # otherwise load transposed
        ozeros = False
        if transposed == True:
            try: 
                o = np.loadtxt(inp.path2qm+"runs/conf_"+str(iconf+1)+"/overlaps/overlap_"+str(jatom+1)+"-"+str(iatom+1)+".dat")
                o = o.reshape(nbas[spe2],nbas[spe1]).T
            except:
                ozeros = True
        # if also the tranposed is not available, then set it as a matrix of zeros
        if ozeros == True:
            o = np.zeros((nbas[spe1],nbas[spe2]),float)
        # single out the overlap dimensions in a multi-dimensional array
        overlap = np.zeros((llmax+1,llmax+1,nnmax,nnmax,2*llmax+1,2*llmax+1))
        n_count1 = {}
        for l1 in range(lmax[spe1]+1):
            n_count1[l1] = 0
        i1 = 0
        for l1 in lvals[spe1]:
            n_count2 = {}
            for l2 in range(lmax[spe2]+1):
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
        for l1 in range(lmax[spe1]+1):
            for n1 in range(nmax[(spe1,l1)]):
                for im1 in range(2*l1+1):
                    isgf2 = iblock2
                    for l2 in range(lmax[spe2]+1):
                        for n2 in range(nmax[(spe2,l2)]):
                            for im2 in range(2*l2+1):
                                Ovlp[isgf1,isgf2] = overlap[l1,l2,n1,n2,im1,im2]
                                isgf2 += 1
                    isgf1 += 1
        iblock2 += nbas[spe2] 
    iblock1 += nbas[spe1] 

print("overlap matrix succesfully filled up")

# Compute density projections on auxiliary functions
Proj = np.dot(Ovlp,Coef)

# Make data directories if not already existing
dirpath = os.path.join(inp.path2qm, "coefficients")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm, "projections")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm, "overlaps")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# Save projections, coefficients and overlaps
np.save(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy",Coef)
np.save(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy",Proj)
np.save(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy",Ovlp)

print((time.time()-start)/60.0, "min")
