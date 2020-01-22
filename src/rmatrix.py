#!/usr/bin/python2.7

import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse
import rmatrix

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-m",   "--msize"  ,     type=int,   default=100, help="number of reference environments")
    parser.add_argument("-rc",   "--cutoffradius"  , type=float, default=4.0, help="soap cutoff")
    parser.add_argument("-sg",   "--sigmasoap"  , type=float, default=0.3, help="soap sigma")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    m = args.msize
    rc = args.cutoffradius
    sg = args.sigmasoap
    return [m,rc,sg]

args = add_command_line_arguments_contraction("density regression")
[M,rc,sigma_soap] = set_variable_values_contraction(args)

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
#================= SOAP PARAMETERS 
zeta = 2.0 
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
nenv = len(spec_array)
#===================== atomic indexes sorted by species
atomicindx = np.zeros((ndata,nspecies,natmax),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[iconf,ispe,icount] = indexes[icount]
#================== species dictionary
spe_dict = {}
spe_dict[0] = "H"
spe_dict[1] = "O"
#====================================== reference environments 
fps_indexes = np.loadtxt("SELECTIONS/refs_selection_"+str(M)+".txt",int)
fps_species = np.loadtxt("SELECTIONS/spec_selection_"+str(M)+".txt",int)
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
#==================================== BASIS SET SIZE ARRAYS
bsize = np.zeros(nspecies,int)
almax = np.zeros(nspecies,int)
anmax = np.zeros((nspecies,llmax+1),int)
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    almax[ispe] = lmax[spe]+1
    for l in xrange(lmax[spe]+1):
        anmax[ispe,l] = nmax[(spe,l)]
        bsize[ispe] += nmax[(spe,l)]*(2*l+1)
#============================================= PROBLEM DIMENSIONALITY 
collsize = np.zeros(M,int)
for iref in xrange(1,M):
    collsize[iref] = collsize[iref-1] + bsize[fps_species[iref-1]]
totsize = collsize[-1] + bsize[fps_species[-1]]
print "problem dimensionality =", totsize
#================================================= TRAINING SETS 
k_MM = np.zeros((llmax+1,M*(2*llmax+1),M*(2*llmax+1)),float) 

for l in xrange(llmax+1):

    power = np.load("POWER_SPECTRA/PS_"+str(l)+".npy")

    if l==0:

        # power spectrum
        nfeat = len(power[0,0])
        power_env = np.zeros((nenv,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata):
            iat = 0
            for ispe in xrange(nspecies):
                for icount in xrange(atom_counting[iconf,ispe]):
                    jat = atomicindx[iconf,ispe,icount]
                    power_per_conf[iconf,jat] = power[iconf,iat]
                    iat+=1
            for iat in xrange(natoms[iconf]):
                power_env[ienv] = power_per_conf[iconf,iat]
                ienv += 1
        power_ref_sparse = power_env[fps_indexes]
        for iref1 in xrange(M):
            for iref2 in xrange(M):
                k_MM[l,iref1,iref2] = np.dot(power_ref_sparse[iref1],power_ref_sparse[iref2].T)**zeta 

    else:

        # power spectrum
        nfeat = len(power[0,0,0])
        power_env = np.zeros((nenv,2*l+1,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,2*l+1,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata):
            iat = 0
            for ispe in xrange(nspecies):
                for icount in xrange(atom_counting[iconf,ispe]):
                    jat = atomicindx[iconf,ispe,icount]
                    power_per_conf[iconf,jat] = power[iconf,iat]
                    iat+=1
            for iat in xrange(natoms[iconf]):
                power_env[ienv] = power_per_conf[iconf,iat]
                ienv += 1
        power_ref_sparse = power_env[fps_indexes]
        power_ref_sparse = power_ref_sparse.reshape(M*(2*l+1),nfeat)
        for iref1 in xrange(M):
            for iref2 in xrange(M):
                k_MM[l,iref1*(2*l+1):iref1*(2*l+1)+2*l+1,iref2*(2*l+1):iref2*(2*l+1)+2*l+1] = np.dot(power_ref_sparse[iref1*(2*l+1):iref1*(2*l+1)+2*l+1],power_ref_sparse[iref2*(2*l+1):iref2*(2*l+1)+2*l+1].T) *  k_MM[0,iref1,iref2]**((zeta-1.0)/zeta) 

Rmat = rmatrix.rmatrix(llmax,nnmax,nspecies,M,totsize,fps_species,almax,anmax,k_MM)

np.save("MATRICES/KMM_"+str(M)+".npy", Rmat)
