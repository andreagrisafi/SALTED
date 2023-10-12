import numpy as np
import sys
import h5py
import os

from salted.sys_utils import read_system, get_atom_idx

def build():

    sys.path.insert(0, './')
    import inp
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    # number of sparse environments
    M = inp.Menv
    zeta = inp.z
    eigcut = inp.eigcut
    sdir = inp.saltedpath+'equirepr_'+inp.saltedname+'/'
    
    kdir = 'kernels_'+inp.saltedname
    if inp.field: kdir += '_field'
    kdir += '/'
    
    # make directories if not exisiting
    dirpath = os.path.join(inp.saltedpath, kdir)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for spe in species:
        for l in range(llmax+1):
            dirpath = os.path.join(inp.saltedpath+kdir, "spe"+str(spe)+"_l"+str(l))
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            dirpath = os.path.join(inp.saltedpath+kdir+"spe"+str(spe)+"_l"+str(l), "M"+str(M)+"_zeta"+str(zeta))
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
    
    kdir = inp.saltedpath+kdir
    
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
    
    # load lambda=0 power spectrum 
    power = h5py.File(sdir+"FEAT-0.h5",'r')['descriptor'][:]
    nfeat = power.shape[-1]
    
    # compute sparse set with FPS
    fps_idx = np.array(do_fps(power.reshape(ndata*natmax,nfeat),M),int)
    fps_species = species_array[fps_idx]
    sparse_set = np.vstack((fps_idx,fps_species)).T
    print("Computed sparse set made of ", M, "environments")
    np.savetxt(sdir+"sparse_set_"+str(M)+".txt",sparse_set,fmt='%i')
    
    # divide sparse set per species
    fps_indexes = {}
    for spe in species:
        fps_indexes[spe] = []
    for iref in range(M):
        fps_indexes[species[fps_species[iref]]].append(fps_idx[iref])
    Mspe = {}
    for spe in species:
        Mspe[spe] = len(fps_indexes[spe])
    
    kernel0_mm = {}
    power_env_sparse = {}
    h5f = h5py.File(sdir+'FEAT-0-M-'+str(M)+'.h5','w')
    if inp.field:
        power2 = h5py.File(sdir+"FEAT-0_field.h5",'r')['descriptor'][:]
        nfeat2 = power2.shape[-1]
        power_env_sparse2 = {}
        h5f2 = h5py.File(sdir+'FEAT-0-M-'+str(M)+'_field.h5','w')
    for spe in species:
        power_env_sparse[spe] = power.reshape(ndata*natmax,nfeat)[np.array(fps_indexes[spe],int)]
        h5f.create_dataset(spe,data=power_env_sparse[spe])
        if inp.field:
            power_env_sparse2[spe] = power2.reshape(ndata*natmax,nfeat2)[np.array(fps_indexes[spe],int)]
            h5f2.create_dataset(spe,data=power_env_sparse2[spe])
    h5f.close()
    if inp.field: h5f2.close()
    
    for spe in species:
        kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
        if inp.field:
            kernel_mm = (np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T) + kernel0_mm[spe]) * kernel0_mm[spe]**(zeta-1)
        else:
            kernel_mm = kernel0_mm[spe]**zeta
       
        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva>eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
        np.save(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy",V)
    
    for l in range(1,llmax+1):
        power = h5py.File(sdir+"FEAT-"+str(l)+".h5",'r')['descriptor'][:]
        nfeat = power.shape[-1]
        power_env_sparse = {}
        h5f = h5py.File(sdir+'FEAT-'+str(l)+'-M-'+str(M)+'.h5','w')
        if inp.field: 
            power2 = h5py.File(sdir+"FEAT-"+str(l)+"_field.h5",'r')['descriptor'][:]
            nfeat2 = power2.shape[-1]
            power_env_sparse2 = {}
            h5f2 = h5py.File(sdir+'FEAT-'+str(l)+'-M-'+str(M)+'_field.h5','w')
        for spe in species:
            power_env_sparse[spe] = power.reshape(ndata*natmax,2*l+1,nfeat)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat)
            h5f.create_dataset(spe,data=power_env_sparse[spe])
            if inp.field:
                power_env_sparse2[spe] = power2.reshape(ndata*natmax,2*l+1,nfeat2)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat2)
                h5f2.create_dataset(spe,data=power_env_sparse2[spe])
        h5f.close()
        if inp.field: h5f2.close()
    
        for spe in species:
            kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
            if inp.field: kernel_mm += np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T)
            for i1 in range(Mspe[spe]):
                for i2 in range(Mspe[spe]):
                    kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
            eva, eve = np.linalg.eigh(kernel_mm)
            eva = eva[eva>eigcut]
            eve = eve[:,-len(eva):]
            V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
            np.save(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy",V)

    return

if __name__ == "__main__":
    build()
