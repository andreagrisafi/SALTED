import os
import sys

import numpy as np

import inp

from salted.sys_utils import read_system

species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

pdir = f"predictions_{inp.saltedname}_{inp.predname}"

M = inp.Menv
ntrain = int(inp.trainfrac*inp.Ntrain)

for i in range(ndata):
    print(f"processing {i+1}/{ndata} frame")
    t = np.load(os.path.join(
        inp.saltedpath, pdir,
        f"M{M}_zeta{inp.z}", f"N{ntrain}_reg{int(np.log10(inp.regul))}",
        f"prediction_conf{i}.npy",
    ))
    n = len(t)

    dirpath = os.path.join(inp.path2qm, inp.predict_data, f"{i+1}")

    idx = np.loadtxt(os.path.join(dirpath, f"idx_prodbas.out")).astype(int)
    idx -= 1

    # # old method
    # idx = list(idx)
    # idx_rev = []
    # for i in range(n):
    #     idx_rev.append(idx.index(i))

    # accelerated method
    idx_rev = np.empty_like(idx)
    idx_rev[idx] = np.arange(len(idx))

    #	np.savetxt('../idx_prodbas_rev.out',idx_rev)
    #	idx_rev = np.loadtxt('../idx_prodbas_rev.out').astype(int)
    cs_list = np.loadtxt(os.path.join(
        dirpath, f"prodbas_condon_shotley_list.out"
    )).astype(int)
    cs_list -= 1

    # # old method
    # cs_list = list(cs_list)
    # t = t[idx_rev]
    # for j in cs_list:
    #     t[j] *= -1

    # accelerated method
    t = t[idx_rev]
    t[cs_list] *= -1

    np.savetxt(os.path.join(dirpath, f"ri_restart_coeffs_predicted.out"), t)

