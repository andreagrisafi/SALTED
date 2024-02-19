import sys
import numpy as np
import time
import h5py

from salted.sys_utils import read_system, get_atom_idx, get_conf_range

def build():

    sys.path.insert(0, './')
    import inp
    
    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
    
    saltedname = inp.saltedname
    
    kdir = inp.saltedpath+"kernels_"+saltedname
    kdir += '/'
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    # number of sparse environments
    M = inp.Menv
    zeta = inp.z
    eigcut = inp.eigcut
    if rank == 0:
        print("M =", M, "eigcut =", eigcut)
        print("zeta =", zeta)
        print("Computing RKHS of symmetry-adapted sparse kernel approximations...")
    sdir = inp.saltedpath+'equirepr_'+inp.saltedname+'/'
    
    # Distribute structures to tasks
    if inp.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = list(range(ndata))
    
    power_env_sparse = {}
    power_env_sparse2 = {}
    kernel0_nm = {}
    
    power2 = h5py.File(sdir+"FEAT-0.h5",'r')["descriptor"][conf_range,:]
    
    for ix in ["x","y","z"]:

        power = h5py.File(sdir+"FEAT-0-"+str(ix)+".h5",'r')["descriptor_"+ix][conf_range,:]
        nfeat = power.shape[-1]
        Mspe = {}
        
        for spe in species:
            if rank == 0: print("lambda = 0", "species:", spe)
            start = time.time()
   
            # Load descriptors and projectors
            power_env_sparse[spe] = h5py.File(sdir+"FEAT-0-M-"+str(M)+"_vfield.h5",'r')[spe][:]
            power_env_sparse2[spe] = h5py.File(sdir+"FEAT-0-M-"+str(M)+".h5",'r')[spe][:]
            Mspe[spe] = power_env_sparse2[spe].shape[0]
            V = np.load(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy")
            
            # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
            for i,iconf in enumerate(conf_range):
                kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
                if zeta>1:
                    kernel0_nm[(iconf,spe)] = np.dot(power2[i,atom_idx[(iconf,spe)]],power_env_sparse2[spe].T)
                    i2x = 0
                    for jx in range(3):
                        #kernel_nm[:,i2x:i2x+Mspe[spe]] *= kernel0_nm[(iconf,spe)]**(zeta-1) 
                        kernel_nm[:,i2x:i2x+Mspe[spe]] *= np.exp(kernel0_nm[(iconf,spe)]) 
                        i2x += Mspe[spe] 
                psi_nm = np.real(np.dot(kernel_nm,V))
                np.save(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm-"+str(ix)+"_conf"+str(iconf)+".npy",psi_nm)
            if rank == 0: print((time.time()-start)/60.0)
        
        # lambda>0
        for l in range(1,llmax+1):
        
            # load power spectrum
            if rank == 0: print("loading lambda =", l)
            power = h5py.File(sdir+"FEAT-"+str(l)+"-"+str(ix)+".h5",'r')["descriptor_"+ix][conf_range,:]
            nfeat = power.shape[-1]
        
            for spe in species:
                if rank == 0: print("lambda = ", l, "species:", spe)
                start = time.time()
        
                # get sparse feature vector for each atomic species
                power_env_sparse[spe] = h5py.File(sdir+"FEAT-"+str(l)+"-M-"+str(M)+"_vfield.h5",'r')[spe][:]
                V = np.load(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy") 
        
                # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
                if zeta==1:
                    for i,iconf in enumerate(conf_range):
                        kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T) 
                        psi_nm = np.real(np.dot(kernel_nm,V))
                        np.save(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm-"+str(ix)+"_conf"+str(iconf)+".npy",psi_nm)
                else:
                    for i,iconf in enumerate(conf_range):
                        kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T)
                        
                        for i1 in range(natom_dict[(iconf,spe)]):
                            i2x = 0
                            for jx in range(3):
                                for i2 in range(Mspe[spe]):
                                    #kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2x*(2*l+1):i2x*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
                                    kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2x*(2*l+1):i2x*(2*l+1)+2*l+1] *= np.exp(kernel0_nm[(iconf,spe)][i1,i2])
                                    i2x += 1
                        psi_nm = np.real(np.dot(kernel_nm,V))
                        np.save(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm-"+str(ix)+"_conf"+str(iconf)+".npy",psi_nm)
                if rank == 0: print((time.time()-start)/60.0)

    return

if __name__ == "__main__":
    build()
