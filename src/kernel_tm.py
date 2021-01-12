import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse

import basis

sys.path.insert(0, './')
import inp

# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i]

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
for spe in spelist:
    llist.append(lmax[spe])
llmax = max(llist)

# read training system
xyzfile = read(inp.path2ref+inp.filename_ref,":")
ndata = len(xyzfile)

# read testing system
xyzfile_testing = read(inp.filename,":")
ndata_testing = len(xyzfile_testing)

# number of sparse environments
M = inp.Menv

zeta = inp.z

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-s",   "--splitting",     type=int, default=1, help="splitting degree")
    parser.add_argument("-p",   "--portion"  ,     type=int, default=1, help="portion of the dataset")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    s = args.splitting
    p = args.portion
    return [s,p]

args = add_command_line_arguments_contraction("density regression")
[nsplit,portion] = set_variable_values_contraction(args)

#dataset_portion = list(np.split(np.array(xrange(ndata),int),nsplit)[portion])
#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)
#======================= system parameters
atomic_symbols_testing = []
atomic_valence_testing = []
natoms_testing = np.zeros(ndata_testing,int)
for i in xrange(len(xyzfile_testing)):
    atomic_symbols_testing.append(xyzfile_testing[i].get_chemical_symbols())
    atomic_valence_testing.append(xyzfile_testing[i].get_atomic_numbers())
    natoms_testing[i] = int(len(atomic_symbols_testing[i]))
natmax_testing = max(natoms_testing)
#==================== species array
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)
spec_list = []
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_valence[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list.append(ispe)
               spec_list_per_conf[iconf].append(ispe)
spec_array = np.asarray(spec_list,int)
nenv = len(spec_array)
#==================== species array
species_testing = np.sort(list(set(np.array([item for sublist in atomic_valence_testing for item in sublist]))))
nspecies_testing = len(species_testing)
spec_list_testing = []
spec_list_per_conf_testing = {}
atom_counting_testing = np.zeros((ndata_testing,nspecies_testing),int)
for iconf in xrange(ndata_testing):
    spec_list_per_conf_testing[iconf] = []
    for iat in xrange(natoms_testing[iconf]):
        for ispe in xrange(nspecies_testing):
            if atomic_valence_testing[iconf][iat] == species_testing[ispe]:
               atom_counting_testing[iconf,ispe] += 1
               spec_list_testing.append(ispe)
               spec_list_per_conf_testing[iconf].append(ispe)
spec_array_testing = np.asarray(spec_list_testing,int)
nenv_testing = len(spec_array_testing)
#===================== atomic indexes sorted by species
atomicindx = np.zeros((ndata,nspecies,natmax),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[iconf,ispe,icount] = indexes[icount]
#===================== atomic indexes sorted by species
atomicindx_testing = np.zeros((ndata_testing,nspecies_testing,natmax_testing),int)
for iconf in xrange(ndata_testing):
    for ispe in xrange(nspecies_testing):
        indexes = [i for i,x in enumerate(spec_list_per_conf_testing[iconf]) if x==ispe]
        for icount in xrange(atom_counting_testing[iconf,ispe]):
            atomicindx_testing[iconf,ispe,icount] = indexes[icount]
#====================================== reference environments 
fps_indexes = np.loadtxt(inp.path2ref+"sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt(inp.path2ref+"sparse_set_"+str(M)+".txt",int)[:,1]
#==================================== BASIS SET SIZE ARRAYS
bsize = np.zeros(nspecies,int)
almax = np.zeros(nspecies,int)
anmax = np.zeros((nspecies,llmax+1),int)
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    almax[ispe] = lmax[spe]+1
    for l in xrange(lmax[spe]+1):
        anmax[ispe,l] = nmax[(spe,l)]
        bsize[ispe] += nmax[(spe,l)]*(2*l+1)
#============================================= PROBLEM DIMENSIONALITY 
collsize = np.zeros(M,int)
for iref in xrange(1,M):
    collsize[iref] = collsize[iref-1] + bsize[fps_species[iref-1]]
totsize = collsize[-1] + bsize[fps_species[-1]]

print "Computing Ktm matrices for each dataset configuration ..."

# load power spectra
power_ref_sparse = {}
for l in xrange(llmax+1):

    power = np.load(inp.path2soap_ref+"SOAP-"+str(l)+".npy")

    if l==0:

        # power spectrum
        nfeat = len(power[0,0])
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
        power_env = np.zeros((nenv,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata):
            #power_per_conf[iconf] = power[iconf]
            iat = 0
            for ispe in xrange(nspecies):
                for icount in xrange(atom_counting[iconf,ispe]):
                    jat = atomicindx[iconf,ispe,icount]
                    power_per_conf[iconf,jat] = power[iconf,iat]
                    iat+=1
            for iat in xrange(natoms[iconf]):
                power_env[ienv] = power_per_conf[iconf,iat]
                ienv += 1
        power_ref_sparse[l] = power_env[fps_indexes]

    else:

        # power spectrum
        nfeat = len(power[0,0,0])
        power_per_conf = np.zeros((ndata,natmax,2*l+1,nfeat),float)
        power_env = np.zeros((nenv,2*l+1,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata):
            #power_per_conf[iconf] = power[iconf]
            iat = 0
            for ispe in xrange(nspecies):
                for icount in xrange(atom_counting[iconf,ispe]):
                    jat = atomicindx[iconf,ispe,icount]
                    power_per_conf[iconf,jat] = power[iconf,iat]
                    iat+=1
            for iat in xrange(natoms[iconf]):
                power_env[ienv] = power_per_conf[iconf,iat]
                ienv += 1
        power_ref_sparse[l] = power_env[fps_indexes]

# testing power spectra
power_testing = {}
for l in xrange(llmax+1):

    power = np.load(inp.path2soap+"SOAP-"+str(l)+".npy")

    if l==0:

        # power spectrum
        nfeat = len(power[0,0])
        power_per_conf = np.zeros((ndata_testing,natmax_testing,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata_testing):
            #power_per_conf[iconf] = power[iconf]
            iat = 0
            for ispe in xrange(nspecies_testing):
                for icount in xrange(atom_counting_testing[iconf,ispe]):
                    jat = atomicindx_testing[iconf,ispe,icount]
                    power_per_conf[iconf,jat] = power[iconf,iat]
                    iat+=1
        power_testing[l] = power_per_conf

    else:

        # power spectrum
        nfeat = len(power[0,0,0])
        power_per_conf = np.zeros((ndata_testing,natmax_testing,2*l+1,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata_testing):
            #power_per_conf[iconf] = power[iconf]
            iat = 0
            for ispe in xrange(nspecies_testing):
                for icount in xrange(atom_counting_testing[iconf,ispe]):
                    jat = atomicindx_testing[iconf,ispe,icount]
                    power_per_conf[iconf,jat] = power[iconf,iat]
                    iat+=1
        power_testing[l] = power_per_conf

startinit = time.time()
# compute sparse kernel matrix
for iconf in xrange(ndata_testing): 
#for iconf in dataset_portion:
    start = time.time()
    atoms = atomic_symbols_testing[iconf]
    # define sparse indexes
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms_testing[iconf],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in xrange(lmax[spe]+1):
            msize = 2*l+1
            for im in xrange(msize):
                for iat in xrange(atom_counting_testing[iconf,ispe]):
                    for imm in xrange(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    # compute kernels
    k_TM = np.zeros(kernel_size,float)
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in xrange(atom_counting_testing[iconf,ispe]):
            iat = atomicindx_testing[iconf,ispe,iatspe]
            ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
            for l in xrange(lmax[spe]+1):
                msize = 2*l+1
                powert = power_testing[l][iconf,iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatspe,l,0,0]
                    k_TM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_TM[ik0]**(float(zeta-1)/zeta)
                    for im1 in xrange(msize):
                        for im2 in xrange(msize):
                            ik = kernel_sparse_indexes[iref,iatspe,l,im1,im2]
                            k_TM[ik] = kern[im2,im1]
    np.savetxt(inp.path2kern+"kernel_conf"+str(iconf)+".dat", k_TM,fmt='%.06e')
#    print iconf, time.time()-start, "seconds"

print iconf+1, "Ktm matrices computed in", (time.time()-startinit)/60.0, "minutes"

