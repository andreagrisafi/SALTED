import numpy as np
import os
import inp
from mpi4py import MPI
from ase.io import read
from sys import argv
from salted.sys_utils import distribute_jobs

# MPI information
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('This is task',rank+1,'of',size,flush=True)
#rank = 0

if (rank == 0):
    if not os.path.exists(inp.path2qm+'coefficients_rho1'):
        os.mkdir(inp.path2qm+'coefficients_rho1')
    if not os.path.exists(inp.path2qm+'coefficients_rho1/x'):
        os.mkdir(inp.path2qm+'coefficients_rho1/x')
    if not os.path.exists(inp.path2qm+'coefficients_rho1/y'):
        os.mkdir(inp.path2qm+'coefficients_rho1/y')
    if not os.path.exists(inp.path2qm+'coefficients_rho1/z'):
        os.mkdir(inp.path2qm+'coefficients_rho1/z')
    if not os.path.exists(inp.path2qm+'projections_rho1'):
        os.mkdir(inp.path2qm+'projections_rho1')
    if not os.path.exists(inp.path2qm+'projections_rho1/x'):
        os.mkdir(inp.path2qm+'projections_rho1/x')
    if not os.path.exists(inp.path2qm+'projections_rho1/y'):
        os.mkdir(inp.path2qm+'projections_rho1/y')
    if not os.path.exists(inp.path2qm+'projections_rho1/z'):
        os.mkdir(inp.path2qm+'projections_rho1/z')

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# Distribute structures to tasks
conf_range = list(range(ndata))
conf_range = distribute_jobs(comm, conf_range)

for i in conf_range:
    dirpath = inp.path2qm+'data/'+str(i+1)+'/'
    #idx = np.loadtxt(dirpath+'idx_prodbas.out').astype(int)
    #cs_list = np.loadtxt(dirpath+'prodbas_condon_shotley_list.out').astype(int)
    #idx -= 1
    #cs_list -= 1
    #idx = list(idx)
    #cs_list = list(cs_list)
    o = np.loadtxt(dirpath+'ri_projections_rho1.out').transpose()
    t = []
    t.append(np.loadtxt(dirpath+'ri_rho1_restart_coeffs_1.out'))
    t.append(np.loadtxt(dirpath+'ri_rho1_restart_coeffs_2.out'))
    t.append(np.loadtxt(dirpath+'ri_rho1_restart_coeffs_3.out'))
    t = np.array(t)
    
    n = len(o)
    
    #for j in cs_list:
    #    o[:,j] *= -1
    #    t[:,j] *= -1
    
    #o = o[:,idx]
    #t = t[:,idx]
    np.save(inp.path2qm+'projections_rho1/x/projections_conf'+str(i)+'.npy',o[0,:])
    np.save(inp.path2qm+'projections_rho1/y/projections_conf'+str(i)+'.npy',o[1,:])
    np.save(inp.path2qm+'projections_rho1/z/projections_conf'+str(i)+'.npy',o[2,:])
    np.save(inp.path2qm+'coefficients_rho1/x/coefficients_conf'+str(i)+'.npy',t[0,:])
    np.save(inp.path2qm+'coefficients_rho1/y/coefficients_conf'+str(i)+'.npy',t[1,:])
    np.save(inp.path2qm+'coefficients_rho1/z/coefficients_conf'+str(i)+'.npy',t[2,:])
