import numpy as np
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

avcoefs = {}
nat_per_species = {}
for spe in spelist:
    nat_per_species[spe] = 0
    avcoefs[spe] = np.zeros(nmax[(spe,0)],float)

print("computing averages...")
for iconf in range(ndata):
    atoms = atomic_symbols[iconf]
    coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    i = 0
    for iat in range(natoms[iconf]):
        spe = atoms[iat] 
        nat_per_species[spe] += 1  
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                for im in range(2*l+1):
                    if l==0:
                       avcoefs[spe][n] += coefs[i]
                    i += 1

for spe in spelist:
    avcoefs[spe] /= nat_per_species[spe]
    np.save("averages_"+str(spe)+".npy",avcoefs[spe])
