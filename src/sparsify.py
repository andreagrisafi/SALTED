import numpy as np
from sys_utils import read_system, get_atom_idx
import sys
sys.path.insert(0, './')
import inp
import argparse
import h5py
import os

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-pr", "--predict", action='store_true', help="Convert descriptors of structures to be predicted")
    parser.add_argument("-c", "--clean", action='store_true', help="Remove old npy files after conversion")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
predict = args.predict
clean = args.clean

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

atom_idx, natom_dict = get_atom_idx(ndata,natoms,spelist,atomic_symbols)

# number of sparse environments
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
if predict:
    sdir = inp.saltedpath+'equirepr_predict_'+inp.saltedname+'/'
else:
    sdir = inp.saltedpath+'equirepr_'+inp.saltedname+'/'
    kdir = inp.saltedpath+'kernels_'+inp.saltedname+'/'

for l in range(llmax+1):
    if os.path.exists(sdir+"FEAT-"+str(l)+".h5"): continue
    a = np.load(sdir+"FEAT-"+str(l)+".npy")
    f = h5py.File(sdir+"FEAT-"+str(l)+".h5",'w')
    f.create_dataset("descriptor",data=a)
    f.close
    if clean: os.remove(sdir+"FEAT-"+str(l)+".npy")

if predict: exit

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
power = h5py.File(sdir+"FEAT-0.h5",'r')['descriptor'][:]
nfeat = power.shape[-1]

if os.path.exists( "sparse_set_"+str(M)+".txt"):
    sparse_set = np.loadtxt("sparse_set_"+str(M)+".txt",dtype=int).T
    fps_idx = sparse_set[0]
    fps_species = sparse_set[1]
else:
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
if not os.path.exists(kdir):
    os.mkdir(kdir)
for spe in spelist:
    for l in range(llmax+1):
        dirpath = os.path.join(kdir, "spe"+str(spe)+"_l"+str(l))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        dirpath = os.path.join(kdir+"spe"+str(spe)+"_l"+str(l), "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

kernel0_mm = {}
power_env_sparse = {}
h5f = h5py.File(sdir+'FEAT-0-M.h5','w')
if inp.field:
    power2 = h5py.File(sdir+"FEAT-0_field.h5",'r')['descriptor'][:]
    power_env_sparse2 = {}
    h5f2 = h5py.File(sdir+'FEAT-0-M_field.h5','w')
for spe in spelist:
    power_env_sparse[spe] = power.reshape(ndata*natmax,nfeat)[np.array(fps_indexes[spe],int)]
    h5f.create_dataset(spe,data=power_env_sparse[spe])
    if inp.field:
        power_env_sparse2[spe] = power2.reshape(ndata*natmax,nfeat)[np.array(fps_indexes[spe],int)]
        h5f.create_dataset(spe,data=power_env_sparse2[spe])
h5f.close()
if inp.field: h5f2.close()

for spe in spelist:
    kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
    # Only one of this and the next if inp.field statements are needed. Need to decide which
    if inp.field:
        kernel_mm = (np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T) + kernel0_mm) * kernel0_mm[spe]**(zeta -1)
    else:
        kernel_mm = kernel0_mm[spe]**zeta
    
    kernel_mm = kernel0_mm[spe]**zeta

    eva, eve = np.linalg.eigh(kernel_mm)
    eva = eva[eva>eigcut]
    eve = eve[:,-len(eva):]
    V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
    np.save(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy",V)

#    if inp.field:
#        kernel_mm = (np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T) + kernel0_mm) * kernel0_mm[spe]**(zeta -1)

#        eva, eve = np.linalg.eigh(kernel_mm)
#        eva = eva[eva>eigcut]
#        eve = eve[:,-len(eva):]
#        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
#        np.save(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector_field.npy",V)

for l in range(1,llmax+1):
    power = h5py.File(sdir+"FEAT-"+str(l)+".h5",'r')['descriptor'][:]
    nfeat = power.shape[-1]
    power_env_sparse = {}
    h5f = h5py.File(sdir+'FEAT-'+str(l)+'-M.h5','w')
    if inp.field: 
        power2 = h5py.File(sdir+"FEAT-"+str(l)+"_field.h5",'r')['descriptor'][:]
        power_env_sparse2 = {}
        h5f2 = h5py.File(sdir+'FEAT-'+str(l)+'-M_field.h5','w')
    for spe in spelist:
        power_env_sparse[spe] = power.reshape(ndata*natmax,2*l+1,nfeat)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat)
        h5f.create_dataset(spe,data=power_env_sparse[spe])
        if inp.field:
            power_env_sparse2[spe] = power2.reshape(ndata*natmax,2*l+1,nfeat)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat)
            h5f2.create_dataset(spe,data=power_env_sparse2[spe])
    h5f.close()
    if inp.field: h5f2.close()

    for spe in spelist:
        kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
        if inp.field: kernel_mm += np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T)
        for i1 in range(Mspe[spe]):
            for i2 in range(Mspe[spe]):
                kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva>eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
        np.save(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/projector.npy",V)
