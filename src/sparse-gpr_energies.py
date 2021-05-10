#!/usr/bin/python

import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
import argparse

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

zeta = inp.z
M = inp.Menv
reg = inp.regul
jit = inp.jitter

nspecies = len(inp.species) 
species = {}
i = 0
for spe in inp.species:
    species[spe] = i
    i+=1

atomic_symbol = []
natoms = np.zeros(ndata,int)
energies = np.zeros(ndata)
stechio = np.zeros((ndata,nspecies),float)
for iconf in xrange(ndata):
    energies[iconf] = xyzfile[iconf].info[inp.propname]
    symbols = xyzfile[iconf].get_chemical_symbols()
    atomic_symbol.append(symbols)
    natoms[iconf] = len(symbols)
    for iat in xrange(natoms[iconf]):
        ispe = species[symbols[iat]]
        stechio[iconf,ispe] += 1.0 
natmax = max(natoms)
covariance = np.dot(stechio.T,stechio)

nostechio = False
if np.linalg.matrix_rank(covariance) < nspecies:
    print "Dataset has uniform distribution of species: no stochiometric baseline is applied."
    nostechio = True
elif nspecies==1:
    print "Dataset has uniform distribution of species: no stochiometric baseline is applied."
    nostechio = True
else:
    print "Dataset has non-uniform distribution of species: a stochiometric baseline is applied."
    vector = np.dot(stechio.T,energies)
    weights = np.linalg.solve(covariance,vector)
    baseline = np.dot(stechio,weights)

print "STD =", np.std(energies), "[energy units]"


#====================================== reference environments 
fps_indexes = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,1] 

spec_list = []
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_symbol[iconf][iat] == inp.species[ispe]:
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

trainrangetot = np.loadtxt("training_set.txt",int)
ntrain = len(trainrangetot)
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
testrange = np.array(np.setdiff1d(range(ndata),trainrangetot),int)
natoms_test = natoms[testrange]
te_energ = energies[testrange]

total_species = np.zeros((ndata,natmax),int)
for iconf in xrange(ndata):
    for iat in xrange(natoms[iconf]):
        total_species[iconf,iat] = spec_list_per_conf[iconf][iat]

power = np.load(inp.path2data+"/soaps/SOAP-0.npy")
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

# kernel NM
k_NM = np.zeros((ndata,M),float)
for iconf in xrange(ndata):
    for iat in xrange(natoms[iconf]):
        for iref in xrange(M):
            k_NM[iconf,iref] += np.dot(power_per_conf[iconf,iat],power_ref_sparse[iref].T)**zeta
    k_NM[iconf,iref] /= natoms[iconf]

# kernel MM
k_MM = np.zeros((M,M),float)
for iref1 in xrange(M):
    for iref2 in xrange(M):
        k_MM[iref1,iref2] = np.dot(power_ref_sparse[iref1],power_ref_sparse[iref2].T)**zeta

for frac in [0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1.0]:
    ntrain = int(frac*len(trainrangetot))
    trainrange = trainrangetot[0:ntrain]
    natoms_train = natoms[trainrange]
   
    ktrain = np.dot(k_NM[trainrange].T,k_NM[trainrange])
    ktest = k_NM[testrange]
  
    if nostechio==False:
        tr_energ = (energies[trainrange] - baseline[trainrange])/natoms_train 
    else:    
        tr_energ = energies[trainrange]/natoms_train - np.mean(energies[trainrange]/natoms_train) 
    
    ml_energ = np.dot(ktest, np.linalg.solve(ktrain + reg*k_MM + jit*np.eye(M), np.dot(k_NM[trainrange].T,tr_energ)))
    if nostechio==False:
        ml_energ *= natoms_test
        ml_energ += baseline[testrange]
    else:
        ml_energ += np.mean(energies[trainrange]/natoms_train)
        ml_energ *= natoms_test
    error = np.sum((ml_energ-te_energ)**2)/float(len(te_energ))
    print "N =", ntrain,"RMSE =", np.sqrt(error) , "[energy units]"
