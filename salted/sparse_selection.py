import numpy as np
import sys
import h5py
import os
import os.path as osp

from salted.sys_utils import ParseConfig, read_system, get_atom_idx, do_fps

def build():
    inp = ParseConfig().parse_input()

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # number of sparse environments
    M, zeta, eigcut = inp.gpr.Menv, inp.gpr.z, inp.gpr.eigcut
    sdir = osp.join(inp.salted.saltedpath, f"equirepr_{inp.salted.saltedname}")

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
    power = h5py.File(osp.join(sdir, "FEAT-0.h5"), 'r')['descriptor'][:]
    nfeat = power.shape[-1]

    # compute sparse set with FPS
    fps_idx = np.array(do_fps(power.reshape(ndata*natmax,nfeat),M),int)
    fps_species = species_array[fps_idx]
    sparse_set = np.vstack((fps_idx,fps_species)).T
    print("Computed sparse set made of ", M, "environments")
    np.savetxt(osp.join(sdir, f"sparse_set_{M}.txt"), sparse_set, fmt='%i')

if __name__ == "__main__":
    build()

