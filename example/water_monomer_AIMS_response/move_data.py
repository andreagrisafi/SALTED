import numpy as np
import os
import inp
from mpi4py import MPI
from ase.io import read

# MPI information
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('This is task',rank+1,'of',size,flush=True)

if (rank == 0):
    if not os.path.exists(inp.path2qm+inp.ovlpdir):
        os.mkdir(inp.path2qm+inp.ovlpdir)
    if not os.path.exists(inp.path2qm+'coefficients/'):
        os.mkdir(inp.path2qm+'coefficients/')
    if not os.path.exists(inp.path2qm+'projections/'):
        os.mkdir(inp.path2qm+'projections/')

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# Distribute structures to tasks
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

for i in conf_range:
    dirpath = inp.path2qm+'data/'+str(i+1)+'/'
    #idx = np.loadtxt(dirpath+'idx_prodbas.out').astype(int)
    #cs_list = np.loadtxt(dirpath+'prodbas_condon_shotley_list.out').astype(int)
    #idx -= 1
    #cs_list -= 1
    #idx = list(idx)
    #cs_list = list(cs_list)
    o = np.loadtxt(dirpath+'ri_projections.out').reshape(-1)
    t = np.loadtxt(dirpath+'ri_restart_coeffs.out').reshape(-1)
    ovlp = np.loadtxt(dirpath+'ri_ovlp.out').reshape(-1)
    
    n = len(o)
    ovlp = ovlp.reshape(n,n)
    
    #for j in cs_list:
    #    ovlp[j,:] *= -1
    #    ovlp[:,j] *= -1
    #    o[j] *= -1
    #    t[j] *= -1
    
    #o = o[idx]
    #t = t[idx]
    #ovlp = ovlp[idx,:]
    #ovlp = ovlp[:,idx]
    np.save(inp.path2qm+inp.ovlpdir+'overlap_conf'+str(i)+'.npy',ovlp)
    np.save(inp.path2qm+'projections/projections_conf'+str(i)+'.npy',o)
    np.save(inp.path2qm+'coefficients/coefficients_conf'+str(i)+'.npy',t)
    os.remove(dirpath+'ri_ovlp.out')
    os.remove(dirpath+'ri_projections.out')
