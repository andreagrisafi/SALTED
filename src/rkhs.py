import os
import numpy as np
import time
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system, get_atom_idx, get_conf_range
import argparse
import h5py

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r", "--response", action='store_true', help="Build kernels using bare lambda=0 projectors introducing non-linearity")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
response = args.response

if inp.parallel:
    from mpi4py import MPI

    # MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size)
else:
    rank = 0

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

atom_idx, natom_dict = get_atom_idx(ndata,natoms,spelist,atomic_symbols)

# number of sparse environments
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
print("M =", M, "eigcut =", eigcut)
print("zeta =", zeta)

sdir = inp.soapdir
kdir = inp.kerndir

# This procedure is done in parallel_rkhs_prep.py for parallel runs
if not inp.parallel:
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
    for spe in spelist:
        species_idx[spe] = ispe
        ispe += 1
    
    species_array = np.zeros((ndata,natmax),int) 
    for iconf in range(ndata):
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            species_array[iconf,iat] = species_idx[spe] 
    species_array = species_array.reshape(ndata*natmax)
    
    # load lambda=0 power spectrum 
    power = np.load(inp.path2ml+sdir+"FEAT-0.npy")
    nfeat = power.shape[-1]
    
    if response:
        # load lambda=0 power spectrum without dummy atom
        power_bare = np.load(inp.path2ml+sdir+"FEAT-0-bare.npy")
        nfeat_bare = power_bare.shape[-1]
    
    # compute sparse set with FPS
    fps_idx = np.array(do_fps(power.reshape(ndata*natmax,nfeat),M),int)
    fps_species = species_array[fps_idx]
    sparse_set = np.vstack((fps_idx,fps_species)).T
    print("Computed sparse set made of ", M, "environments")
    np.savetxt("sparse_set_"+str(M)+".txt",sparse_set,fmt='%i')

    # divide sparse set per species
    fps_indexes = {}
    for spe in spelist:
        fps_indexes[spe] = []
    for iref in range(M):
        fps_indexes[spelist[fps_species[iref]]].append(fps_idx[iref])
    Mspe = {}
    for spe in spelist:
        Mspe[spe] = len(fps_indexes[spe])

   # make directories if not exisiting
    if (rank == 0):
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

print("Computing RKHS of symmetry-adapted sparse kernel approximations...")

# Distribute structures to tasks
if inp.parallel:
    conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
#    if rank == 0:
#        conf_range = [[] for _ in range(size)]
#        blocksize = int(ndata/float(size))
#        for i in range(size):
#            if i == (size-1):
#                conf_range[i] = list(range(ndata))[i*blocksize:ndata]
#            else:
#                conf_range[i] = list(range(ndata))[i*blocksize:(i+1)*blocksize]
#    else:
#        conf_range = None

    conf_range = comm.scatter(conf_range,root=0)
    print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
else:
    conf_range = list(range(ndata))
power_env_sparse = {}
kernel0_mm = {}
kernel0_nm = {}

if response:
    power_env_sparse_bare = {}
    kernel0_mm_temp = {}
    kernel0_nm_temp = {}

if inp.parallel:
    power = h5py.File(inp.path2ml+sdir+"FEAT-0.h5",'r')["descriptor"][conf_range,:]
    nfeat = power.shape[-1]
    Mspe = {} 
    if response:
        power_bare = h5py.File(inp.path2ml+sdir+"FEAT-0-bare.h5",'r')["descriptor"][conf_range,:]
        nfeat_bare = power_bare.shape[-1]

for spe in spelist:
    start = time.time()

    # compute sparse kernel K_MM for each atomic species
    if inp.parallel:
        power_env_sparse[spe] = h5py.File(inp.path2ml+sdir+"FEAT-0-M.h5",'r')[spe][:]
        Mspe[spe] = power_env_sparse[spe].shape[0]
    else:
        power_env_sparse[spe] = power.reshape(ndata*natmax,nfeat)[np.array(fps_indexes[spe],int)]
    kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)

    if response:
        if inp.parallel:
            power_env_sparse_bare[spe] = h5py.File(inp.path2ml+sdir+"FEAT-0-M-bare.h5",'r')[spe][:]
        else:
            power_env_sparse_bare[spe] = power_bare.reshape(ndata*natmax,nfeat_bare)[np.array(fps_indexes[spe],int)]
        kernel0_mm_temp[spe] = kernel0_mm[spe]
        kernel0_mm[spe] = np.dot(power_env_sparse_bare[spe],power_env_sparse_bare[spe].T)
        if not inp.parallel: kernel_mm = kernel0_mm_temp[spe] * kernel0_mm[spe]**(zeta-1)
    else:
        if not inp.parallel: kernel_mm = kernel0_mm[spe]**zeta
    
    if inp.parallel: 
        V = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy")
    else:
    # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva>eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
        np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy",V)

    # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
    for i,iconf in enumerate(conf_range):
        if inp.parallel:
            kernel0_nm[(iconf,spe)] = np.dot(power[i,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
        else:
            kernel0_nm[(iconf,spe)] = np.dot(power[iconf,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)

        if response:
            kernel0_nm_temp[(iconf,spe)] = kernel0_nm[(iconf,spe)]
            if inp.parallel:
                kernel0_nm[(iconf,spe)] = np.dot(power_bare[i,atom_idx[(iconf,spe)]],power_env_sparse_bare[spe].T)
            else:
                kernel0_nm[(iconf,spe)] = np.dot(power_bare[iconf,atom_idx[(iconf,spe)]],power_env_sparse_bare[spe].T)

            kernel_nm = kernel0_nm_temp[(iconf,spe)] * kernel0_nm[(iconf,spe)]**(zeta-1)

        else:
            kernel_nm = kernel0_nm[(iconf,spe)]**zeta

        psi_nm = np.real(np.dot(kernel_nm,V))
        np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
    print((time.time()-start)/60.0)

if response:
    power_env_sparse_bare = {}
    kernel0_mm_temp = {}
    kernel0_nm_temp = {}

# lambda>0
for l in range(1,llmax+1):

    # load power spectrum
    print("loading lambda =", l)
    if inp.parallel:
        power = h5py.File(inp.path2ml+sdir+"FEAT-"+str(l)+".h5",'r')["descriptor"][conf_range,:]
    else:
        power = np.load(inp.path2ml+sdir+"FEAT-"+str(l)+".npy")
    nfeat = power.shape[-1]

    for spe in spelist:
        print("lambda = ", l, "species:", spe)
        start = time.time()

        # get sparse feature vector for each atomic species
        if inp.parallel:
            power_env_sparse[spe] = h5py.File(inp.path2ml+sdir+"FEAT-"+str(l)+"-M.h5",'r')[spe][:]
        else:
            power_env_sparse[spe] = power.reshape(ndata*natmax,2*l+1,nfeat)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat)
        
        # compute K_MM 
        if inp.parallel:
            V = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy")
        else:
            kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T) 
            for i1 in range(Mspe[spe]):
                for i2 in range(Mspe[spe]):
                    kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
    
        # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
            eva, eve = np.linalg.eigh(kernel_mm)
            kernel_mm = None
            eva = eva[eva>eigcut]
            eve = eve[:,-len(eva):]
            V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
            np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy",V)

        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for i,iconf in enumerate(conf_range):
            if inp.parallel:
                kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T) 
            else:
                kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T)

            for i1 in range(natom_dict[(iconf,spe)]):
                for i2 in range(Mspe[spe]):
                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
        print((time.time()-start)/60.0)
