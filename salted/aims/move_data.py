import os
import os.path as osp

import numpy as np
import inp
from ase.io import read

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

if (rank == 0):
    """check if all subdirectories exist, if not create them"""
    sub_dirs = [
        osp.join(inp.saltedpath, d)
        for d in ("overlaps", "coefficients", "projections")
    ]
    for sub_dir in sub_dirs:
        if not osp.exists(sub_dir):
            os.mkdir(sub_dir)

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# Distribute structures to tasks
if inp.parallel:
    if rank == 0:
        conf_range = [[] for _ in range(size)]
        blocksize = int(round(ndata/float(size)))
        for i in range(size):
            if i == (size-1):
                conf_range[i] = list(range(ndata))[i*blocksize:ndata]
            else:
                conf_range[i] = list(range(ndata))[i*blocksize:(i+1)*blocksize]
    else:
        conf_range = None
    conf_range = comm.scatter(conf_range,root=0)
else:
    conf_range = list(range(ndata))

for i in conf_range:
    dirpath = osp.join(inp.path2qm, 'data', str(i+1))
    idx = np.loadtxt(osp.join(dirpath, 'idx_prodbas.out')).astype(int)
    cs_list = np.loadtxt(osp.join(dirpath, 'prodbas_condon_shotley_list.out')).astype(int)
    idx -= 1
    cs_list -= 1
    idx = list(idx)
    cs_list = list(cs_list)
    o = np.loadtxt(osp.join(dirpath, 'ri_projections.out')).reshape(-1)
    t = np.loadtxt(osp.join(dirpath, 'ri_restart_coeffs_df.out')).reshape(-1)
    ovlp = np.loadtxt(osp.join(dirpath, 'ri_ovlp.out')).reshape(-1)

    n = len(o)
    ovlp = ovlp.reshape(n,n)

    for j in cs_list:
        ovlp[j,:] *= -1
        ovlp[:,j] *= -1
        o[j] *= -1
        t[j] *= -1

    o = o[idx]
    t = t[idx]
    ovlp = ovlp[idx,:]
    ovlp = ovlp[:,idx]
    np.save(osp.join(inp.saltedpath, "overlaps", f"overlap_conf{i}.npy"), ovlp)
    np.save(osp.join(inp.saltedpath, "projections", f"projections_conf{i}.npy"), o)
    np.save(osp.join(inp.saltedpath, "coefficients", f"coefficients_conf{i}.npy"), t)

if size > 1: comm.Barrier()


"""delte ri basis overlap and proj coeffs files"""

for i in conf_range:
    dirpath = osp.join(inp.path2qm, 'data', str(i+1))
    os.remove(osp.join(dirpath, 'ri_ovlp.out'))
    os.remove(osp.join(dirpath, 'ri_projections.out'))
