import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse

sys.path.insert(0, '../../lib/')
import rmatrix

sys.path.insert(0, './')
import inp

sys.path.insert(0, '../../src/')
import basis

# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i]

# read basis
[llmax,lmax,nnmax,nmax] = basis.basiset(inp.basis)

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

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
#===================== atomic indexes sorted by species
atomicindx = np.zeros((ndata,nspecies,natmax),int)
for iconf in xrange(ndata):
    for ispe in xrange(nspecies):
        indexes = [i for i,x in enumerate(spec_list_per_conf[iconf]) if x==ispe]
        for icount in xrange(atom_counting[iconf,ispe]):
            atomicindx[iconf,ispe,icount] = indexes[icount]
#====================================== reference environments 
fps_indexes = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt("sparse_set_"+str(M)+".txt",int)[:,1]
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

print "Computing Knm matrices for each dataset configuration ..."

# load power spectra
power_ref_sparse = {}
power_training = {}
for l in xrange(llmax+1):

    power = np.load("SOAP-"+str(l)+".npy")

    if l==0:

        # power spectrum
        nfeat = len(power[0,0])
        power_env = np.zeros((nenv,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata):
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
        power_training[l] = power_per_conf

    else:

        # power spectrum
        nfeat = len(power[0,0,0])
        power_env = np.zeros((nenv,2*l+1,nfeat),float)
        power_per_conf = np.zeros((ndata,natmax,2*l+1,nfeat),float)
        ienv = 0
        for iconf in xrange(ndata):
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
        power_training[l] = power_per_conf

startinit = time.time()
# compute sparse kernel matrix
for iconf in xrange(ndata): 
#for iconf in dataset_portion:
    start = time.time()
    atoms = atomic_symbols[iconf]
    # define sparse indexes
    kernel_size = 0
    kernel_sparse_indexes = np.zeros((M,natoms[iconf],llmax+1,2*llmax+1,2*llmax+1),int)
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for l in xrange(lmax[spe]+1):
            msize = 2*l+1
            for im in xrange(msize):
                for iat in xrange(atom_counting[iconf,ispe]):
                    for imm in xrange(msize):
                        kernel_sparse_indexes[iref,iat,l,im,imm] = kernel_size
                        kernel_size += 1
    # compute kernels
    k_NM = np.zeros(kernel_size,float)
    for iref in xrange(M):
        ispe = fps_species[iref]
        spe = spe_dict[ispe]
        for iatspe in xrange(atom_counting[iconf,ispe]):
            iat = atomicindx[iconf,ispe,iatspe]
            ik0 = kernel_sparse_indexes[iref,iatspe,0,0,0]
            for l in xrange(lmax[spe]+1):
                msize = 2*l+1
                powert = power_training[l][iconf,iat]
                powerr = power_ref_sparse[l][iref]
                if l==0:
                    ik = kernel_sparse_indexes[iref,iatspe,l,0,0]
                    k_NM[ik] = np.dot(powert,powerr)**zeta
                else:
                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
                    for im1 in xrange(msize):
                        for im2 in xrange(msize):
                            ik = kernel_sparse_indexes[iref,iatspe,l,im1,im2]
                            k_NM[ik] = kern[im2,im1]
    np.savetxt("kernels/kernel_conf"+str(iconf)+".dat", k_NM,fmt='%.06e')
#    print iconf, time.time()-start, "seconds"

print iconf, "Knm matrices computed in", (time.time()-startinit)/60.0, "minutes"

