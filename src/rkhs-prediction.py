import os
import numpy as np
import time
from ase.io import read
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system, get_atom_idx

if inp.parallel:
    from mpi4py import MPI

    # MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size)
else:
    rank = 0

spelist, lmax, nmax, llmax, nnmax, ndata_train, atomic_symbols_train, natoms_train, natmax_train = read_system()

# read system test set
xyzfile = read(inp.predict_filename,":")
ndata = len(xyzfile)

# number of sparse environments
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
print("M =", M, "eigcut =", eigcut)
print("zeta =", zeta)

sdirtrain = inp.soapdir
kdirtrain = inp.kerndir
sdir = inp.predict_soapdir
kdir = inp.predict_kerndir

# system parameters test set
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in range(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

# load lambda=0 power spectrum 
power_train = np.load(inp.path2ml+sdirtrain+"FEAT-0.npy")
nfeat_train = power_train.shape[-1]
power = np.load(inp.path2ml+sdir+"FEAT-0.npy")
nfeat = power.shape[-1]

# compute sparse set with FPS
fps_idx = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,1]

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

atom_idx, natom_dict = get_atom_idx(ndata,natoms,spelist,atomic_symbols)

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

# Distribute structures to tasks
if inp.parallel:
    if rank == 0:
        conf_range = [[] for _ in range(size)]
        blocksize = int(round(ndata/float(size)))
        for i in range(size):
            if i == (size-1):
                conf_range[i] = list(range(ndata))[i*blocksize:ndata]
            else:
                conf_range[i] = list(range(ndata))[i*blocksize:(i+1)*blocksize]
    else:
        conf_range = None

    conf_range = comm.scatter(conf_range,root=0)
    print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
else:
    conf_range = list(range(ndata))

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
    V = np.load(inp.path2ml+kdirtrain+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy")

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
    power_train = np.load(inp.path2ml+sdirtrain+"FEAT-"+str(l)+".npy")
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
        V = np.load(inp.path2ml+kdirtrain+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy")

        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for iconf in range(ndata):
            kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T) 
            for i1 in range(natom_dict[(iconf,spe)]):
                for i2 in range(Mspe[spe]):
                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
        print((time.time()-start)/60.0)
