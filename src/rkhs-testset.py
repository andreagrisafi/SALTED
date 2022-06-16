import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle

import basis

sys.path.insert(0, './')
import inp

spelist = inp.species

# read system training set
xyzfile_train = read(inp.path2ref+inp.filename,":")
ndata_train = len(xyzfile_train)

# read system test set
xyzfile = read(inp.filenametest,":")
ndata = len(xyzfile)

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
for spe in spelist:
    llist.append(lmax[spe])
llmax = max(llist)

# number of sparse environments
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
print("M =", M, "eigcut =", eigcut)
print("zeta =", zeta)

sdirtrain = inp.soapdir
sdir = inp.soapdirtest
kdir = inp.kerndirtest

# system parameters training set
atomic_symbols_train = []
natoms_train = np.zeros(ndata_train,int)
for i in range(ndata_train):
    atomic_symbols_train.append(xyzfile_train[i].get_chemical_symbols())
    natoms_train[i] = int(len(atomic_symbols_train[i]))
natmax_train = max(natoms_train)

# system parameters test set
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in range(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

# load lambda=0 power spectrum 
power_train = np.load(inp.path2mlref+sdirtrain+"FEAT-0.npy")
nfeat_train = power_train.shape[-1]
power = np.load(inp.path2ml+sdir+"FEAT-0.npy")
nfeat = power.shape[-1]

# compute sparse set with FPS
fps_idx = np.loadtxt(inp.path2ref+"sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt(inp.path2ref+"sparse_set_"+str(M)+".txt",int)[:,1]

# make directories if not exisiting
dirpath = os.path.join(inp.path2ml, kdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
for spe in spelist:
    for l in range(llmax+1):
        dirpath = os.path.join(inp.path2ml+kdir, "spe"+str(spe)+"_l"+str(l))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        dirpath = os.path.join(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l), "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

# initialize useful arrays
atom_idx = {}
natom_dict = {}
for iconf in range(ndata):
    for spe in spelist:
        atom_idx[(iconf,spe)] = [] 
        natom_dict[(iconf,spe)] = 0 

# extract species-dependent power spectrum for lambda=0
for iconf in range(ndata):
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        atom_idx[(iconf,spe)].append(iat)
        natom_dict[(iconf,spe)] += 1 

# divide sparse set per species
fps_indexes = {}
for spe in spelist:
    fps_indexes[spe] = []
for iref in range(M):
    fps_indexes[spelist[fps_species[iref]]].append(fps_idx[iref])
Mspe = {}
for spe in spelist:
    Mspe[spe] = len(fps_indexes[spe])

print("Computing RKHS of symmetry-adapted sparse kernel approximations...")

# lambda = 0
power_env_sparse = {}
kernel0_mm = {}
kernel0_nm = {}
for spe in spelist:
    print("lambda = 0", "species:", spe)
    start = time.time()

    # compute sparse kernel K_MM for each atomic species 
    power_env_sparse[spe] = power_train.reshape(ndata_train*natmax_train,power_train.shape[-1])[np.array(fps_indexes[spe],int)]
    kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
    kernel_mm = kernel0_mm[spe]**zeta
    
    # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
    eva, eve = np.linalg.eigh(kernel_mm)
    eva = eva[eva>eigcut]
    eve = eve[:,-len(eva):]
    V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))

    # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
    for iconf in range(ndata):
        kernel0_nm[(iconf,spe)] = np.dot(power[iconf,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
        kernel_nm = kernel0_nm[(iconf,spe)]**zeta
        psi_nm = np.real(np.dot(kernel_nm,V))
        np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
    print((time.time()-start)/60.0)

# lambda>0
for l in range(1,llmax+1):

    # load power spectrum
    print("loading lambda =", l)
    power_train = np.load(inp.path2mlref+sdirtrain+"FEAT-"+str(l)+".npy")
    nfeat_train = power_train.shape[-1]
    power = np.load(inp.path2ml+sdir+"FEAT-"+str(l)+".npy")
    nfeat = power.shape[-1]

    power_env_sparse = {}
    for spe in spelist:
        print("lambda = ", l, "species:", spe)
        start = time.time()

        # get sparse feature vector for each atomic species
        power_env_sparse[spe] = power_train.reshape(ndata_train*natmax_train,2*l+1,nfeat_train)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat_train)
        
        # compute K_MM 
        kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T) 
        for i1 in range(Mspe[spe]):
            for i2 in range(Mspe[spe]):
                kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
    
        # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva>eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))

        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for iconf in range(ndata):
            kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T) 
            for i1 in range(natom_dict[(iconf,spe)]):
                for i2 in range(Mspe[spe]):
                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
        print((time.time()-start)/60.0)
