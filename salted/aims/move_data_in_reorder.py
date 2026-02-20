import os
import sys

import numpy as np
from salted.sys_utils import ParseConfig, read_system, distribute_jobs

def build():
    inp = ParseConfig().parse_input()

    print("WARNING! This script assumes you will use an AIMS version < 240403 to read the predicted RI coefficients. If this is not true, please use move_data_in instead.")

    if inp.system.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print('This is task',rank+1,'of',size,flush=True)
    else:
        comm = None
        rank = 0
        size = 1
    
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    pdir = f"predictions_{inp.salted.saltedname}_{inp.prediction.predname}"
    
    ntrain = int(inp.gpr.trainfrac * inp.gpr.Ntrain)
    
    # Distribute structures to tasks
    if inp.system.parallel:
        conf_range = distribute_jobs(comm, list(range(ndata)))
    else:
        conf_range = list(range(ndata))
    
    for i in conf_range:
        print(f"processing {i+1}/{ndata} frame")
        t = np.load(os.path.join(
            inp.salted.saltedpath, pdir,
            f"M{inp.gpr.Menv}_zeta{inp.gpr.z}", f"N{ntrain}_reg{int(np.log10(inp.gpr.regul))}",
            f"prediction_conf{i}.npy",
        ))
        n = len(t)
    
        dirpath = os.path.join(inp.qm.path2qm, inp.prediction.predict_data, f"{i+1}")
    
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
