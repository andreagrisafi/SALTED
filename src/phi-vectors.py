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
# read system
xyzfile = read(inp.filename,":")
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

print "Computing RKHS of sparse GPR..."
print ""

#def do_fps(x, d=0):
#    # Code from Giulio Imbalzano
#    if d == 0 : d = len(x)
#    n = len(x)
#    iy = np.zeros(d,int)
#    iy[0] = 0
#    # Faster evaluation of Euclidean distance
#    n2 = np.sum((x*np.conj(x)),axis=1)
#    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
#    for i in xrange(1,d):
#        iy[i] = np.argmax(dl)
#        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
#        dl = np.minimum(dl,nd)
#    return iy

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))

# compute number of atomic environments for each species
nenv = {}
for spe in spelist:
    nenv[spe] = 0

for iconf in xrange(ndata):
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        nenv[spe] += 1

# load lambda=0 power spectrum
power = np.load(inp.path2data+"soaps/FEAT-0.npy")
nfeat = power.shape[-1]

power_env = {}
ispe = {}
for spe in spelist:
    power_env[spe] = np.zeros((nenv[spe],nfeat))
    ispe[spe] = 0
    for l in xrange(llmax+1):
        dirpath = os.path.join(inp.path2data+"kernels", "spe"+str(spe)+"_l"+str(l))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
    
atom_idx = {}
natom_dict = {}
for iconf in xrange(ndata):
    for spe in spelist:
        atom_idx[(iconf,spe)] = [] 
        natom_dict[(iconf,spe)] = 0 

# extract species-dependent power spectrum for lambda=0
for iconf in xrange(ndata):
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        power_env[spe][ispe[spe]] = power[iconf,iat]
        ispe[spe] += 1
        atom_idx[(iconf,spe)].append(iat)
        natom_dict[(iconf,spe)] += 1 

# load sparse set of environments
fps_environ = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,1]

# divide sparse set per species
fps_indexes = {}
for spe in spelist:
    fps_indexes[spe] = []
for iref in xrange(M):
    fps_indexes[spelist[fps_species[iref]]].append(fps_environ[iref])
Mspe = {}
for spe in spelist:
    Mspe[spe] = len(fps_indexes[spe])

# lambda = 0
power_env_sparse = {}
kernel0_mm = {}
kernel0_nm = {}
for spe in spelist:

    # compute sparse kernel K_MM for each atomic species 
    power_env_sparse[spe] = power.reshape(ndata*natoms[0],power.shape[-1])[np.array(fps_indexes[spe],int)]
    kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
    kernel_mm = kernel0_mm[spe]**zeta
    
    # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
    eva, eve = np.linalg.eigh(kernel_mm)
    eva = eva[eva>inp.eigcut]
    eve = eve[:,-len(eva):]
    V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))

    # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
    for iconf in xrange(ndata):
        kernel0_nm[(iconf,spe)] = np.dot(power[iconf,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
        kernel_nm = kernel0_nm[(iconf,spe)]**zeta
        psi_nm = np.real(np.dot(kernel_nm,V))
        np.save(inp.path2data+"kernels/spe"+str(spe)+"_l"+str(0)+"/psi-nm_conf"+str(iconf)+"_M"+str(M)+".npy",psi_nm)

# lambda>0
for l in xrange(1,llmax+1):

    # load power spectrum
    power = np.load(inp.path2data+"soaps/FEAT-"+str(l)+".npy")
    nfeat = power.shape[-1]

    power_env = {}
    ispe = {}
    for spe in spelist:
        power_env[spe] = np.zeros((nenv[spe],2*l+1,nfeat))
        ispe[spe] = 0

    # extract species-dependent power spectrum from lambda=0 FPS
    for iconf in xrange(ndata):
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            power_env[spe][ispe[spe]] = power[iconf,iat]
            ispe[spe] += 1

    power_env_sparse = {}
    for spe in spelist:

        # get sparse feature vector for each atomic species
        power_env_sparse[spe] = power.reshape(ndata*natoms[0],2*l+1,power.shape[-1])[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat)
        
        # compute K_MM 
        kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T) 
        for i1 in xrange(Mspe[spe]):
            for i2 in xrange(Mspe[spe]):
                kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
    
        # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva>inp.eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))

        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for iconf in xrange(ndata):
            kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T) 
            for i1 in xrange(natom_dict[(iconf,spe)]):
                for i2 in xrange(Mspe[spe]):
                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(inp.path2data+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(iconf)+"_M"+str(M)+".npy",psi_nm)
