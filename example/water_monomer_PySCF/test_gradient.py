import sys
import time
import numpy as np
from ase.io import read
from salted import init_pred, sph_utils
from salted import salted_prediction
from salted.sys_utils import ParseConfig, detect_mpi
import matplotlib.pyplot as plt

inp = ParseConfig().parse_input()

comm, size, rank, _ = detect_mpi()

# Initialize SALTED prediction
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals = init_pred.build(rank)
# Truncated angular order
lcut = 6

d = np.array([0.1, 0.01, 0.001])
icart = {}
icart["x"] = 0
icart["y"] = 1
icart["z"] = 2

# Select Cartesian axis for gradient
axis = "y"
d_c = icart[axis]

# Select configuration
icfg = -1 
structure = read("./water_total.xyz", ":")[icfg]
atomic_symbols = structure.get_chemical_symbols()
natoms = len(structure.positions)

# Select displaced atom for the gradient
iat = 0

coefs = {}
grad_finite_diff = {}

# Analytical gradient
output = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,size,rank,lcut,True,structure) 
grad_pred_coefs = output[1]

# Compute gradient by finite-differences for each atomic displacement d
for i in range(len(d)):

    # +d
    structure.positions[iat,d_c] += d[i]
    output = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,size,rank,lcut,False,structure)
    coefs[str(d[i])] = output[0].copy()

    # -d 
    structure.positions[iat,d_c] -= 2*d[i]
    output = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,size,rank,lcut,False,structure)
    coefs[str("m"+str(d[i]))] = output[0].copy()
    
    # Compute gradient
    grad_finite_diff[str(d[i])] = (np.array(coefs[str(d[i])]) - np.array(coefs[str("m"+str(d[i]))]))/(2*d[i])
    
    # Restore atomic positions
    structure.positions[iat,d_c] += d[i]

if rank == 0:
    
    # Plot results for a given chemical species and radial channel 
    species = "O"
    irad = 0

    for lam in range(5):

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
            for l in range(min(lmax[spe],lcut)+1):
                for n in range(nmax[(spe,l)]):
                    if (spe == species) and (l == lam) and (n==irad):
                        for im in range(2*l+1):
                            array[im].append(grad_pred_coefs[iat,d_c,itot+im])
                            for k in range(len(d)):
                                array_fd[im][k].append(grad_finite_diff[str(d[k])][itot+im])
                    itot += 2*l+1
        
        for im in range(2*lam+1):
            print([np.mean(np.abs(np.array(array_fd[im][k])-np.array(array[im]))) for k in range(len(d))])
            plt.loglog(1/d, [np.mean(np.abs(np.array(array_fd[im][k])-np.array(array[im]))) for k in range(len(d))], "-", label = r"$\mu =$"+ str(im-lam))
        
        plt.xlabel("Delta "+axis + r" [$A^{-1}$]")
        plt.ylabel("Finite-difference error [arb. units.]")
        plt.title(r"$\nabla_{"+axis+",atom-"+str(iat)+"}$, lam=" + str(lam)+", spe="+species+", n="+ str(irad))
        plt.legend()
        plt.savefig("lam-"+str(lam)+".png")
        plt.clf()

