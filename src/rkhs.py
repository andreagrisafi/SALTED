import os
import sys
import numpy as np
import time
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system, get_atom_idx, get_conf_range
import h5py

saltedname = inp.saltedname

if inp.field==True:
    kdir = "kernels_"+saltedname+"_field"
else:
    kdir = "kernels_"+saltedname

species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

# number of sparse environments
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
print("M =", M, "eigcut =", eigcut)
print("zeta =", zeta)

print("Computing RKHS of symmetry-adapted sparse kernel approximations...")

# Distribute structures to tasks
if inp.parallel:
    if rank == 0:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
#        conf_range = [[] for _ in range(size)]
#        blocksize = int(round(ndata/float(size)))
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
for spe in species:
    print("lambda = 0", "species:", spe)
    start = time.time()

    # compute sparse kernel K_MM for each atomic species 
    power_env_sparse = power.reshape(ndata*natmax,power.shape[-1])[np.array(fps_indexes[spe],int)]
    if inp.field:
        power_env_sparse2 = power2.reshape(ndata*natmax,power2.shape[-1])[np.array(fps_indexes[spe],int)]
        kernel_mm = np.dot(power_env_sparse,power_env_sparse.T)
        kernel0_mm[spe] = np.dot(power_env_sparse2,power_env_sparse2.T)
        kernel_mm += kernel0_mm[spe]
        kernel_mm *= kernel0_mm[spe]**(zeta-1)
    else:
        kernel0_mm[spe] = np.dot(power_env_sparse,power_env_sparse.T)
        kernel_mm = kernel0_mm[spe]**zeta
    
    # compute RKHS of K_MM^-1 cutting small/negative eigenvalues
    eva, eve = np.linalg.eigh(kernel_mm)
    eva = eva[eva>eigcut]
    eve = eve[:,-len(eva):]
    V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
    np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy",V)

    # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
    for iconf in range(ndata):
        if inp.field:
            kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]],power_env_sparse.T)
            kernel0_nm[(iconf,spe)] = np.dot(power2[iconf,atom_idx[(iconf,spe)]],power_env_sparse2.T)
            kernel_nm += kernel0_nm[(iconf,spe)]
            kernel_nm *= kernel0_nm[(iconf,spe)]**(zeta-1)
        else:
            kernel0_nm[(iconf,spe)] = np.dot(power[iconf,atom_idx[(iconf,spe)]],power_env_sparse.T)
            kernel_nm = kernel0_nm[(iconf,spe)]**zeta
        psi_nm = np.real(np.dot(kernel_nm,V))
        np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
    print((time.time()-start)/60.0)

# lambda>0
for l in range(1,llmax+1):

    # load power spectrum
    print("loading lambda =", l)
    if inp.field:
        power = np.load(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(l)+"_field.npy")
        power2 = np.load(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(l)+".npy")
    else:
        power = np.load(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(l)+".npy")

    for spe in species:
        print("lambda = ", l, "species:", spe)
        start = time.time()

        # get sparse feature vector for each atomic species
        power_env_sparse = power.reshape(ndata*natmax,2*l+1,power.shape[-1])[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),power.shape[-1])
        if inp.field:
            power_env_sparse2 = power2.reshape(ndata*natmax,2*l+1,power2.shape[-1])[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),power2.shape[-1])
        
        # compute K_MM 
        kernel_mm = np.dot(power_env_sparse,power_env_sparse.T) 
        if inp.field:
            kernel_mm += np.dot(power_env_sparse2,power_env_sparse2.T) 
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
            kernel_nm = np.dot(power[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),power.shape[-1]),power_env_sparse.T) 
            if inp.field:
                kernel_nm += np.dot(power2[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),power2.shape[-1]),power_env_sparse2.T) 
            for i1 in range(natom_dict[(iconf,spe)]):
                for i2 in range(Mspe[spe]):
                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
        print((time.time()-start)/60.0)
