import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle

sys.path.insert(0, './')
import inp

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# number of sparse environments
M = inp.Menv

print "Computing a sparse set made of", M, "FPS environments..."

def do_fps(x, d=0):
    # Code from Giulio Imbalzano
    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum((x*np.conj(x)),axis=1)
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
    for i in xrange(1,d):
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int)
for i in xrange(ndata):
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
#====================== environmental power spectrum
power = np.load(inp.path2ml+"soaps/SOAP-0.npy")
nfeat = len(power[0,0])
power_env = np.zeros((nenv,nfeat),complex)
ienv = 0
for iconf in xrange(ndata):
    power_per_conf = np.zeros((natoms[iconf],nfeat),complex)
    iat = 0
    for ispe in xrange(nspecies):
        for icount in xrange(atom_counting[iconf,ispe]):
            jat = atomicindx[iconf,ispe,icount] 
            power_per_conf[jat,:] = power[iconf,iat,:]
            iat+=1
    for iat in xrange(natoms[iconf]):
        power_env[ienv,:] = power_per_conf[iat,:]
        ienv += 1 

fps_indexes = np.array(do_fps(power_env,M),int)
fps_species = spec_array[fps_indexes]
sparse_set = np.vstack((fps_indexes,fps_species)).T
np.savetxt("sparse_set_"+str(M)+".txt",sparse_set,fmt='%i')
