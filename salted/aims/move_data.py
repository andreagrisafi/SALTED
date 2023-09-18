import numpy as np
import os
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
    if not os.path.exists(inp.saltedpath+"overlaps"):
        os.mkdir(inp.saltedpath+"overlaps")
    if not os.path.exists(inp.saltedpath+"coefficients"):
        os.mkdir(inp.saltedpath+"coefficients")
    if not os.path.exists(inp.saltedpath+"projections"):
        os.mkdir(inp.saltedpath+"projections")

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
    dirpath = inp.path2qm+'data/'+str(i+1)+'/'
    idx = np.loadtxt(dirpath+'idx_prodbas.out').astype(int)
    cs_list = np.loadtxt(dirpath+'prodbas_condon_shotley_list.out').astype(int)
    idx -= 1
    cs_list -= 1
    idx = list(idx)
    cs_list = list(cs_list)
    o = np.loadtxt(dirpath+'ri_projections.out').reshape(-1)
    t = np.loadtxt(dirpath+'ri_restart_coeffs_df.out').reshape(-1)
    ovlp = np.loadtxt(dirpath+'ri_ovlp.out').reshape(-1)
    
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
    np.save(inp.saltedpath+'overlaps/overlap_conf'+str(i)+'.npy',ovlp)
    np.save(inp.saltedpath+'projections/projections_conf'+str(i)+'.npy',o)
    np.save(inp.saltedpath+'coefficients/coefficients_conf'+str(i)+'.npy',t)

if size > 1: comm.Barrier()

for i in conf_range:
    dirpath = inp.path2qm+'data/'+str(i+1)+'/'
    os.remove(dirpath+'ri_ovlp.out')
    os.remove(dirpath+'ri_projections.out')
