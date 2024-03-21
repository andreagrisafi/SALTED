import os
import sys

import numpy as np
import inp
from salted.sys_utils import read_system, get_conf_range

def build():
    
    print("WARNING! This script assumes you will use an AIMS version < 2403XX to read the predicted RI coefficients. If this is not true, please use move_data_in instead.")

    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print('This is task',rank+1,'of',size,flush=True)
    else:
        rank = 0
        size = 1
    
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    pdir = f"predictions_{inp.saltedname}_{inp.predname}"
    
    M = inp.Menv
    ntrain = int(inp.trainfrac*inp.Ntrain)
    
    # Distribute structures to tasks
    if inp.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
    else:
        conf_range = list(range(ndata))
    
    for i in conf_range:
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
   
        # accelerated method
        idx_rev = np.empty_like(idx)
        idx_rev[idx] = np.arange(len(idx))
   
        cs_list = np.loadtxt(os.path.join(
            dirpath, f"prodbas_condon_shotley_list.out"
        )).astype(int)
        cs_list -= 1
   
        # accelerated method
        t = t[idx_rev]
        t[cs_list] *= -1
    
        np.savetxt(os.path.join(dirpath, f"ri_restart_coeffs_predicted.out"), t)

if __name__ == "__main__":
    build()
