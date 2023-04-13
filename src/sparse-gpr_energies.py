#!/usr/bin/python

import numpy as np
import sys
sys.path.insert(0, './')
import inp
from ase.io import read
from sys_utils import read_system

spelist, lmax, nmax, llmax, nnmax, ndata, symbols, natoms, natmax = read_system()

xyzfile = read(inp.filename,":")

zeta = inp.z
M = inp.Menv
reg = inp.regul
jit = 1e-8
#jit = inp.jitter

nspecies = len(inp.species) 
species = {}
i = 0
for spe in inp.species:
    species[spe] = i
    i+=1

energies = np.zeros(ndata)
stechio = np.zeros((ndata,nspecies),float)
for iconf in range(ndata):
    energies[iconf] = xyzfile[iconf].info[inp.propname]
    for iat in range(natoms[iconf]):
        ispe = species[symbols[iconf][iat]]
        stechio[iconf,ispe] += 1.0 
covariance = np.dot(stechio.T,stechio)

nostechio = False
if np.linalg.matrix_rank(covariance) < nspecies:
    print("Dataset has uniform distribution of species: no stochiometric baseline is applied.")
    nostechio = True
elif nspecies==1:
    print("Dataset has uniform distribution of species: no stochiometric baseline is applied.")
    nostechio = True
else:
    print("Dataset has non-uniform distribution of species: a stochiometric baseline is applied.")
    vector = np.dot(stechio.T,energies)
    weights = np.linalg.solve(covariance,vector)
    baseline = np.dot(stechio,weights)

print("STD =", np.std(energies), "[energy units]")

# load reference environments 
fps_indexes = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]

# load training set and define test set accordingly
trainrangetot = np.loadtxt("training_set.txt",int)
ntrain = len(trainrangetot)
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
testrange = np.array(np.setdiff1d(range(ndata),trainrangetot),int)
natoms_test = natoms[testrange]
te_energ = energies[testrange]

# load feature vector and define sparse feature vector
power_per_conf = np.load(inp.path2ml+inp.soapdir+"FEAT-0.npy")
nfeat = power_per_conf.shape[-1]
power_ref_sparse = power_per_conf.reshape(ndata*3,nfeat)[fps_indexes]

# compute kernel that couples N structure with M environments
k_NM = np.zeros((ndata,M),float)
for iconf in range(ndata):
    for iref in range(M):
        for iat in range(natoms[iconf]):
            k_NM[iconf,iref] += np.dot(power_per_conf[iconf,iat],power_ref_sparse[iref].T)**zeta
        k_NM[iconf,iref] /= natoms[iconf]

# DECOMMENT BELOW FOR FULL GPR
#k_NN = np.zeros((ndata,ndata),float)
#for iconf in range(ndata):
#    for jconf in range(ndata):
#        for iat in range(natoms[iconf]):
#            for jat in range(natoms[jconf]):
#                k_NN[iconf,jconf] += np.dot(power_per_conf[iconf,iat],power_per_conf[jconf,jat].T)**zeta

# compute environmental kernel for M environments
k_MM = np.zeros((M,M),float)
for iref1 in range(M):
    for iref2 in range(M):
        k_MM[iref1,iref2] = np.dot(power_ref_sparse[iref1],power_ref_sparse[iref2].T)**zeta

# compute the RKHS of K_MM^-1 cutting small/negative eigenvalues
eva, eve = np.linalg.eigh(k_MM)
eva = eva[::-1]
eve = eve[:,::-1]
eva = eva[eva>inp.eigcut]
Mcut = len(eva)
eve = eve[:,:Mcut]
V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))

# span training set fraction
for frac in [0.025,0.05,0.1,0.2,0.4,1.0]:

    # define training set
    ntrain = int(frac*len(trainrangetot))
    trainrange = trainrangetot[0:ntrain]
    natoms_train = natoms[trainrange]
   
    # compute the RKHS of K_NM * K_MM^-1 * K_MN^T
    Phi = np.dot(k_NM[trainrange],V)
    Phi_test = np.dot(k_NM[testrange],V)
    
    # DECOMMENT BELOW FOR FULL GPR
    #ktrain = k_NN[trainrange][:,trainrange]
    #ktest = k_NN[testrange][:,trainrange]
   
    # baseline property 
    if nostechio==False:
        tr_energ = (energies[trainrange] - baseline[trainrange])/natoms_train 
    else:    
        tr_energ = (energies[trainrange]/natoms_train - np.mean(energies[trainrange]/natoms_train))
    
    # DECOMMENT BELOW FOR FULL GPR
    #ml_energ = np.dot(ktest, np.linalg.solve(ktrain + reg*np.eye(ntrain), tr_energ))
   
    # DECOMMENT FOR UNSTABLE SPARSE GPR
    #ml_energ = np.dot(ktest, np.linalg.solve(ktrain + reg*k_MM + jit*np.eye(M), np.dot(k_NM[trainrange].T,tr_energ)))

    # perform regression and prediction
    ml_energ = np.dot(Phi_test, np.linalg.solve(np.dot(Phi.T,Phi) + reg*np.eye(Mcut), np.dot(Phi.T,tr_energ)))

    # reconstruct property
    if nostechio==False:
        ml_energ *= natoms_test
        ml_energ += baseline[testrange]
    else:
        ml_energ += np.mean(energies[trainrange]/natoms_train)
        ml_energ *= natoms_test
    
    # compute prediction error
    error = np.sum((ml_energ-te_energ)**2)/float(len(te_energ))
    print("N =", ntrain,"RMSE =", np.sqrt(error) , "[energy units]")
