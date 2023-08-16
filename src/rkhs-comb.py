import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle

from sys_utils import read_system, get_atom_idx
import basis

sys.path.insert(0, './')
import inp


species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

########################################################################################

for iconf in range(ndata):
    # Define relevant species
    excluded_species = []
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))

# recompute number of atoms
natoms_total = 0
natoms_list = []
natoms = np.zeros(ndata,int)
for iconf in range(ndata):
    natoms[iconf] = 0
    for spe in species:
        natoms[iconf] += natom_dict[(iconf,spe)]
    natoms_total += natoms[iconf]
    natoms_list.append(natoms[iconf])
natmax = max(natoms_list)

# recompute atomic indexes from new species selections
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

#############################################################################

# number of sparse environments
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
print("M =", M, "eigcut =", eigcut)
print("zeta =", zeta)

kdir = "kernels_"+inp.saltedname+"_"+inp.saltedname2

def do_fps(x, d=0):
    # FPS code from Giulio Imbalzano
    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum((x*np.conj(x)),axis=1)
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
    for i in range(1,d):
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

# compute number of atomic environments for each species
ispe = 0
species_idx = {}
for spe in species:
    species_idx[spe] = ispe
    ispe += 1

species_array = np.zeros((ndata,natmax),int) 
for iconf in range(ndata):
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        species_array[iconf,iat] = species_idx[spe] 
species_array = species_array.reshape(ndata*natmax)

# make directories if not exisiting
dirpath = os.path.join(inp.saltedpath, kdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
for spe in species:
    for l in range(llmax+1):
        dirpath = os.path.join(inp.saltedpath+kdir+"/", "spe"+str(spe)+"_l"+str(l))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        dirpath = os.path.join(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l), "M"+str(M)+"_zeta"+str(zeta))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

# load lambda=0 power spectrum 
power = np.load(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-0.npy")
power2 = np.load(inp.saltedpath+"equirepr_"+inp.saltedname2+"/FEAT-0.npy")
nfeat1 = power.shape[-1]
nfeat2 = power2.shape[-1]

# compute sparse set with FPS
fps_idx = np.array(do_fps(power2.reshape(ndata*natmax,nfeat2),M),int)
fps_species = species_array[fps_idx]
sparse_set = np.vstack((fps_idx,fps_species)).T
print("Computed sparse set made of ", M, "environments")
np.savetxt(inp.saltedpath+"equirepr_"+inp.saltedname+"/sparse_set_"+str(M)+".txt",sparse_set,fmt='%i')

# initialize useful arrays
atom_idx = {}
natom_dict = {}
for iconf in range(ndata):
    for spe in species:
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
for spe in species:
    fps_indexes[spe] = []
for iref in range(M):
    fps_indexes[species[fps_species[iref]]].append(fps_idx[iref])
Mspe = {}
for spe in species:
    Mspe[spe] = len(fps_indexes[spe])

print("Computing RKHS of symmetry-adapted sparse kernel approximations...")

# lambda = 0
power_env_sparse = {}
power_env_sparse2 = {}
kernel0_mm = {}
kernel0_nm = {}
for spe in species:
    print("lambda = 0", "species:", spe)
    start = time.time()

    # compute sparse kernel K_MM for each atomic species 
    power_env_sparse[spe] = power.reshape(ndata*natmax,power.shape[-1])[np.array(fps_indexes[spe],int)]
    power_env_sparse2[spe] = power2.reshape(ndata*natmax,power2.shape[-1])[np.array(fps_indexes[spe],int)]
    kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
    kernel0_mm[spe] += np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T)
    kernel_mm = kernel0_mm[spe]**zeta
    
    # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
    eva, eve = np.linalg.eigh(kernel_mm)
    eva = eva[eva>eigcut]
    eve = eve[:,-len(eva):]
    V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
    np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy",V)

    # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
    for iconf in range(ndata):
        kernel0_nm[(iconf,spe)] = np.dot(power[iconf,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
        kernel0_nm[(iconf,spe)] += np.dot(power2[iconf,atom_idx[(iconf,spe)]],power_env_sparse2[spe].T)
        kernel_nm = kernel0_nm[(iconf,spe)]**zeta
        psi_nm = np.real(np.dot(kernel_nm,V))
        np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
    print((time.time()-start)/60.0)

# lambda>0
for l in range(1,llmax+1):

    # load power spectrum
    print("loading lambda =", l)
    power = np.load(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(l)+".npy")
    power2 = np.load(inp.saltedpath+"equirepr_"+inp.saltedname2+"/FEAT-"+str(l)+".npy")
    nfeat1 = power.shape[-1]
    nfeat2 = power2.shape[-1]

    power_env_sparse = {}
    power_env_sparse2 = {}
    for spe in species:
        print("lambda = ", l, "species:", spe)
        start = time.time()

        # get sparse feature vector for each atomic species
        power_env_sparse[spe] = power.reshape(ndata*natmax,2*l+1,nfeat1)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat1)
        power_env_sparse2[spe] = power2.reshape(ndata*natmax,2*l+1,nfeat2)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat2)
        
        # compute K_MM 
        kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T) 
        kernel_mm += np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T) 
        for i1 in range(Mspe[spe]):
            for i2 in range(Mspe[spe]):
                kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
    
        # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva>eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
        np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy",V)

        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for iconf in range(ndata):
            kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat1),power_env_sparse[spe].T) 
            kernel_nm += np.dot(power2[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat2),power_env_sparse2[spe].T) 
            for i1 in range(natom_dict[(iconf,spe)]):
                for i2 in range(Mspe[spe]):
                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
        print((time.time()-start)/60.0)
