import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read

import basis
sys.path.insert(0, './')
import inp

# read species
species = inp.species

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)


# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int) 
for i in range(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

projector = {}
ncut = {}
for spe in species:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]

avcoefs = {}
nat_per_species = {}
for spe in species:
    nat_per_species[spe] = 0
    avcoefs[spe] = np.zeros(nmax[(spe,0)],float)

print("computing averages...")
for iconf in range(ndata):
    atoms = atomic_symbols[iconf]
    coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
    i = 0
    for iat in range(natoms[iconf]):
        spe = atoms[iat] 
        nat_per_species[spe] += 1  
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                for im in range(2*l+1):
                    if l==0:
                       avcoefs[spe][n] += coefs[i]
                    i += 1

for spe in species:
    avcoefs[spe] /= nat_per_species[spe]
    np.save("averages_"+str(spe)+".npy",avcoefs[spe])
