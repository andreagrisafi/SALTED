import os
import sys
import time
import numpy as np
from ase.io import read
from salted import init_pred 
from salted import salted_prediction 

from salted.sys_utils import ParseConfig, detect_mpi
inp = ParseConfig().parse_input()

comm, size, rank, _ = detect_mpi()

(saltedname, saltedpath, saltedtype,
filename, species, average,
path2qm, qmcode, qmbasis, dfbasis,
filename_pred, predname, predict_data, alpha_only,
rep1, rcut1, sig1, nrad1, nang1, neighspe1,
rep2, rcut2, sig2, nrad2, nang2, neighspe2,
sparsify, nsamples, ncut,
zeta, Menv, Ntrain, trainfrac, regul, eigcut,
gradtol, restart, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

ntrain = int(Ntrain*trainfrac)
vdir = f"validations_{saltedname}"
reg_log10_intstr = str(int(np.log10(regul)))
dirpath = os.path.join(saltedpath, vdir, f"M{Menv}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}")

# Initialize SALTED prediction
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals = init_pred.build(rank)

lcut = 6
gradient=False

# do prediction for the given structure    
frames = read(inp.prediction.filename,":")
for i in range(len(frames)):
    structure = frames[i]
    [coefs] = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,dipole_integrals,comm,size,rank,lcut,gradient,structure) 
    if rank==0:
        ref_coefs = np.loadtxt(dirpath+"/COEFFS-"+str(Ntrain+i+1)+".dat")
        print("Conf", i+1, "Consistent prediction?", np.allclose(coefs,ref_coefs)) 
