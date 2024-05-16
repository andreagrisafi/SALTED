import os
import sys
import numpy as np
import os.path as osp

from salted.sys_utils import ParseConfig, read_system

def build():

    inp = ParseConfig().parse_input()

    spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

    avcoefs = {}
    nat_per_species = {}
    for spe in spelist:
        nat_per_species[spe] = 0
        avcoefs[spe] = np.zeros(nmax[(spe,0)],float)

    print("computing averages...")
    for iconf in range(ndata):
        atoms = atomic_symbols[iconf]
        coefs = np.load(os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"))
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

    adir = os.path.join(inp.salted.saltedpath, "coefficients", "averages")
    if not osp.exists(adir):
        os.mkdir(adir)

    for spe in spelist:
        avcoefs[spe] /= nat_per_species[spe]
        np.save(os.path.join(inp.salted.saltedpath, "coefficients", "averages", f"averages_{spe}.npy"), avcoefs[spe])

    return

if __name__ == "__main__":
    build()
