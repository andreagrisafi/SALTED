import sys
import time
import numpy as np
from ase.io import read
from salted import init_pred, sph_utils
from salted import salted_prediction, salted_prediction_gradient
from salted.sys_utils import ParseConfig

inp = ParseConfig().parse_input()

if inp.system.parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    fcomm = comm.py2f()
    rank = comm.Get_rank()
    ntasks = comm.Get_size()
    if rank==0: print("Parallel run over", ntasks, "tasks",flush=True)

else:
    comm = None
    rank = 0
    ntasks = 1

d = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])

bohr2angs = 0.529177249

icfg = 0
structure = read("../water_total.xyz", ":")[icfg]
natoms = len(structure.positions)
d_c = 0
axis = "x"
iat = 0
lam = 3
species = "O"
channel = 0

# Initialize SALTED prediction
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals = init_pred.build()
 
ave = {}
ave_grad = {}
grad_fd = {}

# Initialize electrode charge density 
output = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,ntasks,rank,True,structure) 
ave["ref"] = output[0]
grad_pred_coefs = output[1]

for i in range(len(d)):
    structure.positions[iat,d_c] += d[i]
    output = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,ntasks,rank,False,structure)
    ave[str(d[i])] = output[0].copy()

    structure.positions[iat,d_c] -= 2*d[i]
    output = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,ntasks,rank,False,structure)
    ave[str("m"+str(d[i]))] = output[0].copy()
    
    structure.positions[iat,d_c] += d[i]
    
    grad_fd[str(d[i])] = (np.array(ave[str(d[i])]) - np.array(ave[str("m"+str(d[i]))]))/(2*d[i])

if rank == 0:
    atomic_symbols = structure.get_chemical_symbols()
    import matplotlib.pyplot as plt
    plt.clf()
    
    array = []
    array_fd = []
    for im in range(2*lam+1):
        array.append([])
        array_fd.append([])
        for k in range(len(d)):
            array_fd[im].append([])
    
    itot = 0
    for iatc in range(natoms):
        spe = atomic_symbols[iatc]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                if (spe == species) and (l == lam) and (n==channel):
                    for im in range(2*l+1):
                        array[im].append(grad_pred_coefs[iat,d_c,itot+(im)])
                        for k in range(len(d)):
                            array_fd[im][k].append(grad_fd[str(d[k])][itot+(im)])
                itot += (2*l+1)
    
    for im in range(2*lam+1):
        plt.loglog(1/d, [np.mean(np.abs(np.array(array_fd[im][k])-np.array(array[im]))) for k in range(len(d))], "-", label = r"$\mu =$"+ str(im-lam))
        if np.mean(np.abs(np.array(array_fd[im][4])-np.array(array[im])))> 10**(-9):
            print(str(im))
    
    plt.xlabel("d"+axis + r" [$A^{-1}$]")
    plt.ylabel("abs error")
    plt.title(r"$\nabla_{x,0,O}$, z=1, lam=" + str(lam)+", spe=O, n="+ str(channel))
    plt.legend()
    plt.savefig("./dcoeffnotavez1"+axis+str(lam)+species+str(channel)+".png")
    plt.clf()

