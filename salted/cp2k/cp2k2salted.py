import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
import copy
import time

from salted import basis

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
[lmax,nmax] = basis.basiset(inp.dfbasis)

#dirpath = os.path.join(inp.saltedpath, "coefficients-nofield")
dirpath = os.path.join(inp.saltedpath, "coefficients-efield")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# init geometry
for iconf in range(0,57):
    geom = xyzfile[iconf]
    symbols = geom.get_chemical_symbols()
    natoms = len(symbols)
    # compute basis set size
    nRI = 0
    for iat in range(natoms):
        spe = symbols[iat]
        if spe in species:
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    nRI += 2*l+1

    # load density coefficients and check dimension
    #coefficients = np.loadtxt(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.coeffile)
    coefficients = np.loadtxt(inp.path2qm+"conf_"+str(iconf+1)+"/efield/"+inp.coeffile)
    if len(coefficients)!=nRI:
        print("ERROR: basis set size does not correspond to size of coefficients vector!")
        sys.exit(0)
    else:
        print("conf", iconf+1, "size =", nRI, flush=True)
    
    # save coefficients vector in SALTED format
    if natoms%2 != 0:
        coefficients = np.sum(coefficients,axis=1)   
    #np.save(inp.saltedpath+"coefficients-nofield/coefficients_conf"+str(iconf)+".npy",coefficients)
    np.save(inp.saltedpath+"coefficients-efield/coefficients_conf"+str(iconf)+".npy",coefficients)

#    # save overlap matrix in SALTED format
#    overlap = np.zeros((nRI, nRI)).astype(np.double)
#    for i in range(nRI):
#        offset = 4 + i*((nRI+1)*8)
#        overlap[:, i] = np.fromfile(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.ovlpfile, dtype=np.float64, offset = offset, count=nRI)
#    
#    dirpath = os.path.join(inp.saltedpath, "overlaps")
#    if not os.path.exists(dirpath):
#       os.mkdir(dirpath)
#    np.save(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy",overlap)
    
    ## save projections vector in SALTED format
    #projections = np.dot(overlap,coefficients)
    #dirpath = os.path.join(inp.saltedpath, "projections")
    #if not os.path.exists(dirpath):
    #    os.mkdir(dirpath)
    #np.save(inp.saltedpath+"projections/projections_conf"+str(iconf)+".npy",projections)
