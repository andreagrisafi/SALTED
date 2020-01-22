import sys
import numpy as np
import scipy
from scipy import special
import math
from math import factorial
import time
import ase
from ase import io
from ase.io import read

bohr2ang = 0.529177249
#========================== system definition
filename = "examples/water_monomer.xyz"
xyzfile = read(filename,":")
ndata = len(xyzfile)
#======================= system parameters
coords = []
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int) 
for i in xrange(len(xyzfile)):
    coords.append(np.asarray(xyzfile[i].get_positions(),float)/bohr2ang)
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

#================== species dictionary
spe_dict = {}
spe_dict[0] = "H"
spe_dict[1] = "O"
#============== angular 
lmax = {}
llmax = 3
lmax["O"] = 3
lmax["H"] = 2
nnmax = 10
nmax = {}
# oxygen
nmax[("O",0)] = 10
nmax[("O",1)] = 7
nmax[("O",2)] = 5
nmax[("O",3)] = 2
# hydrogen
nmax[("H",0)] = 4
nmax[("H",1)] = 3
nmax[("H",2)] = 2

#===================================================== start decomposition
for iconf in xrange(ndata):
    start = time.time() 
    print "-------------------------------"  
    print "iconf = ", iconf
    coordinates = coords[iconf]
    atoms = atomic_symbols[iconf]
    valences = atomic_valence[iconf]
    #==================================================
    totsize = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            totsize += nmax[(atoms[iat],l)]*(2*l+1)
    #==================================================
    coeffs = np.loadtxt("COEFFICIENTS/coord_"+str(iconf).zfill(3)+".dat")
    overlap = np.loadtxt("OVERLAPS/coord_"+str(iconf).zfill(3)+".dat")
    #==================================================
    Coef = np.zeros(totsize,float)
    Over = np.zeros((totsize,totsize),float)
    i1 = 0
    for iat in xrange(natoms[iconf]):
        spe1 = atoms[iat] 
        for l1 in xrange(lmax[spe1]+1):
            for n1 in xrange(nmax[(spe1,l1)]):
                for im1 in xrange(2*l1+1):
                    #
                    if l1==1 and im1!=2:
                        Coef[i1] = coeffs[i1+1] 
                    elif l1==1 and im1==2:
                        Coef[i1] = coeffs[i1-2] 
                    else:
                        Coef[i1] = coeffs[i1] 
                    #
                    i2 = 0
                    for jat in xrange(natoms[iconf]):
                        spe2 = atoms[jat] 
                        for l2 in xrange(lmax[spe2]+1):
                            for n2 in xrange(nmax[(spe2,l2)]):
                                for im2 in xrange(2*l2+1):
                                    #
                                    if l1==1 and im1!=2 and l2!=1:
                                        Over[i1,i2] = overlap[i1+1,i2] 
                                    elif l1==1 and im1==2 and l2!=1:
                                        Over[i1,i2] = overlap[i1-2,i2]
                                    #
                                    elif l2==1 and im2!=2 and l1!=1:
                                        Over[i1,i2] = overlap[i1,i2+1] 
                                    elif l2==1 and im2==2 and l1!=1:
                                        Over[i1,i2] = overlap[i1,i2-2] 
                                    #
                                    elif l1==1 and im1!=2 and l2==1 and im2!=2:
                                        Over[i1,i2] = overlap[i1+1,i2+1] 
                                    elif l1==1 and im1!=2 and l2==1 and im2==2:
                                        Over[i1,i2] = overlap[i1+1,i2-2] 
                                    elif l1==1 and im1==2 and l2==1 and im2!=2:
                                        Over[i1,i2] = overlap[i1-2,i2+1] 
                                    elif l1==1 and im1==2 and l2==1 and im2==2:
                                        Over[i1,i2] = overlap[i1-2,i2-2] 
                                    #
                                    else:
                                        Over[i1,i2] = overlap[i1,i2]
                                    i2 += 1
                    i1 += 1

    Proj = np.dot(Over,Coef)
    ########################################################################################################
    np.save("PROJS_NPY/projections_conf"+str(iconf)+".npy",Proj)
    np.save("OVER_NPY/overlap_conf"+str(iconf)+".npy",Over)
    print "time =", time.time()-start 
