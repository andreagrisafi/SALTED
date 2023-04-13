import basis
import sys
sys.path.insert(0, './')
import inp
import numpy as np
from ase.io import read

def read_system(filename=inp.filename):
    
    # read species
    spelist = inp.species
    
    # read basis
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    llist = []
    nlist = []
    for spe in spelist:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    nnmax = max(nlist)
    llmax = max(llist)
    
    # read system
    xyzfile = read(filename,":")
    ndata = len(xyzfile)
    
    #======================= system parameters
    atomic_symbols = []
    natoms = np.zeros(ndata,int) 
    for i in range(len(xyzfile)):
        atomic_symbols.append(xyzfile[i].get_chemical_symbols())
        natoms[i] = int(len(atomic_symbols[i]))
    natmax = max(natoms)

    return spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols,natoms, natmax

def get_atom_idx(ndata,natoms,spelist,atomic_symbols):
    # initialize useful arrays
    atom_idx = {}
    natom_dict = {}
    for iconf in range(ndata):
        for spe in spelist:
            atom_idx[(iconf,spe)] = [] 
            natom_dict[(iconf,spe)] = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            atom_idx[(iconf,spe)].append(iat)
            natom_dict[(iconf,spe)] += 1 

    return atom_idx,natom_dict
