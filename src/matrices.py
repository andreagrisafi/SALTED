import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle
from scipy import sparse

from sys_utils import read_system,get_atom_idx
import basis

sys.path.insert(0, './')
import inp

# system definition
spelist = inp.species
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# basis definition
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

coefdir = inp.coefdir

species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)


###############################################################################################

natoms_total = 0
natoms_list = []
natoms = np.zeros(ndata,int)
for iconf in range(ndata):
    natoms[iconf] = 0
    for spe in species:
        natoms[iconf] += natoms_per_spe[(iconf,spe)]
    natoms_total += natoms[iconf]
    natoms_list.append(natoms[iconf])
    # Define excluded species
    excluded_species = []
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))
natmax = max(natoms_list)

# recompute atomic indexes from new species selections
atoms_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

###############################################################################################

p = sparse.load_npz(inp.saltedpath+"rkhs-vectors_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf0.npz")
totsize = p.shape[-1]
print("problem dimensionality:", totsize,flush=True)
if totsize>70000:
    print("ERROR: problem dimension too large, minimize directly loss-function instead!")
    sys.exit(0)

if inp.average:
    # load average density coefficients
    av_coefs = {}
    for spe in spelist:
        av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

dirpath = os.path.join(inp.saltedpath, "regrdir_"+inp.saltedname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.saltedpath+"regrdir_"+inp.saltedname+"/", "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# define training set at random
dataset = list(range(ndata))
#random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt(inp.saltedpath+"regrdir_"+inp.saltedname+"/training_set_N"+str(inp.Ntrain)+".txt",trainrangetot,fmt='%i')

# Distribute structures to tasks
ntrain = int(inp.trainfrac*inp.Ntrain)
trainrange = trainrangetot[:ntrain]

print("computing regression matrices...")
Avec = np.zeros(totsize)
Bmat = np.zeros((totsize,totsize))
#Avec = np.load(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N240.npy")*240
#Bmat = np.load(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N240.npy")*240
for iconf in trainrange:
    print("conf:", iconf+1,flush=True)
    
    # load reference QM data
    ref_coefs = np.load(inp.saltedpath+coefdir+"coefficients_conf"+str(iconf)+".npy")
    over = np.load(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy")
    psivec = sparse.load_npz(inp.saltedpath+"rkhs-vectors_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz")
    psi = psivec.toarray()

    if inp.average:

        # fill array of average spherical components
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            if spe in species:
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if l==0:
                           Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1
        
        # subtract average
        ref_coefs -= Av_coeffs
    
    ref_projs = np.dot(over,ref_coefs)
    
    Avec += np.dot(psi.T,ref_projs)
    Bmat += np.dot(psi.T,np.dot(over,psi))

    if iconf+1==160 or iconf+1==200 or iconf+1==220: 
        np.save(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(iconf+1)+".npy",Avec/float(iconf+1))
        np.save(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(iconf+1)+".npy",Bmat/float(iconf+1))

Avec /= float(ntrain)
Bmat /= float(ntrain)

np.save(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(ntrain)+".npy",Avec)
np.save(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(ntrain)+".npy",Bmat)
