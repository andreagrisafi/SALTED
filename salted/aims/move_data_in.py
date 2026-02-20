import os
import sys

import numpy as np

from salted.sys_utils import ParseConfig, read_system, distribute_jobs


def build():
    inp = ParseConfig().parse_input()

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
    
    if rank == 0: print("WARNING! This script assumes you will use an AIMS version >= 240403 to read the predicted RI coefficients. If this is not true, please use move_data_in_reorder instead.")

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(filename=inp.prediction.filename,spelist = inp.system.species, dfbasis = inp.qm.dfbasis)
    
    pdir = f"predictions_{inp.salted.saltedname}_{inp.prediction.predname}"
    
    ntrain = int(inp.gpr.trainfrac*inp.gpr.Ntrain)
    
    # Distribute structures to tasks
    if inp.system.parallel:
        conf_range = distribute_jobs(comm, list(range(ndata)))
    else:
        conf_range = list(range(ndata))
    
    for i in conf_range:
        print(f"processing {i+1}/{ndata} frame")
        t = np.loadtxt(os.path.join(
            inp.salted.saltedpath, pdir,
            f"M{inp.gpr.Menv}_zeta{inp.gpr.z}", f"N{ntrain}_reg{int(np.log10(inp.gpr.regul))}",
            f"COEFFS-{i+1}.dat",
        ))
        n = len(t)
    
        dirpath = os.path.join(inp.qm.path2qm, inp.prediction.predict_data, f"{i+1}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    
        np.savetxt(os.path.join(dirpath, f"ri_restart_coeffs_predicted.out"), t)

if __name__ == "__main__":
    build()
