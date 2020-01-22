#!/usr/bin/python

import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-m",   "--msize"  ,     type=int,   default=100, help="number of reference environments")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    m = args.msize
    return [m]

args = add_command_line_arguments_contraction("density regression")
[M] = set_variable_values_contraction(args)

def do_fps(x, d=0):
    # Code from Giulio Imbalzano
    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum((x*np.conj(x)),axis=1)
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
    for i in xrange(1,d):
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

#========================== system definition
filename = "coords_1000.xyz"
xyzfile = read(filename,":")
ndata = len(xyzfile)
#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)
#==================== species array
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
print species

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
#====================== environmental power spectrum
power = np.load("POWER_SPECTRA/PS_0.npy")
nfeat = len(power[0,0])
power_env = np.zeros((nenv,nfeat),complex)
ienv = 0
for iconf in xrange(ndata):
    power_per_conf = np.zeros((natoms[iconf],nfeat),complex)
    iat = 0
    for ispe in xrange(nspecies):
        for icount in xrange(atom_counting[iconf,ispe]):
            jat = atomicindx[iconf,ispe,icount] 
            power_per_conf[jat,:] = power[iconf,iat,:]
            iat+=1
    for iat in xrange(natoms[iconf]):
        power_env[ienv,:] = power_per_conf[iat,:]
        ienv += 1 

fps_indexes = np.array(do_fps(power_env,M),int)
fps_species = spec_array[fps_indexes]
np.savetxt("SELECTIONS/refs_selection_"+str(M)+".txt",fps_indexes,fmt='%i')
np.savetxt("SELECTIONS/spec_selection_"+str(M)+".txt",fps_species,fmt='%i')
