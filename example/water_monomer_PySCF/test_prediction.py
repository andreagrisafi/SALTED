import os
import sys
import time
import numpy as np
from ase.io import read
from salted import init_pred 
from salted import salted_prediction 

from salted.sys_utils import ParseConfig, detect_mpi
inp = ParseConfig().parse_input()

comm, size, rank, parallel = detect_mpi()

ntrain = round(inp.gpr.Ntrain*inp.gpr.trainfrac)
vdir = f"validations_{inp.salted.saltedname}"
reg_log10_intstr = str(int(np.log10(inp.gpr.regul)))
dirpath = os.path.join(inp.salted.saltedpath, vdir, f"M{inp.gpr.Menv}_zeta{inp.gpr.z}", f"N{ntrain}_reg{reg_log10_intstr}")

# Initialize SALTED prediction
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals = init_pred.build(rank)

lcut = 6
gradient=False

# do prediction for the given structure    
frames = read(inp.prediction.filename,":")
for i in range(len(frames)):
    structure = frames[i]
    [coefs] = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,size,rank,lcut,gradient,structure)
    if parallel:
        comm.Barrier()
        coefs = comm.allreduce(coefs)
    if rank==0:
        ref_coefs = np.loadtxt(dirpath+"/COEFFS-"+str(inp.gpr.Ntrain+i+1)+".dat")
        print("Conf", i+1, "Consistent prediction?", np.allclose(coefs,ref_coefs)) 
