#!/usr/bin/python

import sys
import numpy as np
import scipy
from scipy import special
import time
import ase
from ase import io
from ase.io import read
import argparse

bohr2ang = 0.529177249
#========================== system definition
filename = "coords_1000.xyz"
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
#==================== species array
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)
spec_list = []
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_valence[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list.append(ispe)
               spec_list_per_conf[iconf].append(ispe)
spec_array = np.asarray(spec_list,int)

spe_dict = {}
spe_dict[0] = "H" 
spe_dict[1] = "O"

nenv = {}
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    nenv[spe] = 0
    for iconf in xrange(ndata):
        nenv[spe] += atom_counting[iconf,ispe]
    print spe, nenv[spe]

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

av_coefs = {}
for spe in ["H","O"]:
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)

print "computing averages..."
for iconf in xrange(ndata):
    atoms = atomic_symbols[iconf]
    #==================================================
    Proj = np.load("PROJS_NPY/projections_conf"+str(iconf)+".npy")
    Over = np.load("OVER_NPY/overlap_conf"+str(iconf)+".npy")
    Coef = np.linalg.solve(Over,Proj)
    #==================================================
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat] 
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       av_coefs[spe][n] += Coef[i]
                    i += 1

print "saving averages..."
for spe in ["H","O"]:
    av_coefs[spe] /= nenv[spe]
    np.save("AVERAGES/"+str(spe)+".npy",av_coefs[spe])

print "computing baselined projections..."
for iconf in xrange(ndata):
    atoms = atomic_symbols[iconf]
    #==================================================
    totsize = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            totsize += nmax[(atoms[iat],l)]*(2*l+1)
    #==================================================
    Proj = np.load("PROJS_NPY/projections_conf"+str(iconf)+".npy")
    Over = np.load("OVER_NPY/overlap_conf"+str(iconf)+".npy")
    #==================================================
    Av_coeffs = np.zeros(totsize,float)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n] 
                    i += 1
    #==================================================
    Proj -= np.dot(Over,Av_coeffs)
    np.savetxt("BASELINED_PROJECTIONS/projections_conf"+str(iconf)+".dat",Proj, fmt='%.10e')
    np.savetxt("OVER_DAT/overlap_conf"+str(iconf)+".dat", np.concatenate(Over), fmt='%.10e')


