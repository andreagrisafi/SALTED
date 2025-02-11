import os
import sys
import time
import os.path as osp

import h5py
import numpy as np
from scipy import special

from salted import basis
from salted.sys_utils import ParseConfig, get_feats_projs

def build():

    inp = ParseConfig().parse_input()

    saltedname = inp.salted.saltedname
    saltedpath = inp.salted.saltedpath
    species = inp.system.species
    Menv = inp.gpr.Menv
    zeta = inp.gpr.z
    reg = inp.gpr.regul
    ncut = inp.descriptor.sparsify.ncut
    sparsify = True if inp.descriptor.sparsify.ncut > 0 else False
  
    # read basis
    [lmax,nmax] = basis.basiset(inp.qm.dfbasis)
    llist = []
    nlist = []
    for spe in species:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    lmax_max = max(llist)

    charge_integrals = {}
    if inp.qm.qmcode=="cp2k":
        # Initialize calculation of density/density-response moments
        from salted.cp2k.utils import init_moments 
        charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,0)

    loadstart = time.time()
   
    # Load feature space sparsification information if required 
    if sparsify:
        vfps = {}
        for lam in range(lmax_max+1):
            vfps[lam] = np.load(osp.join(
                saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    # Load training feature vectors and RKHS projection matrix
    Vmat,Mspe,power_env_sparse = get_feats_projs(species,lmax)
 
    # load regression weights
    ntrain = int(inp.gpr.Ntrain*inp.gpr.trainfrac)
    weights = np.load(osp.join(
        saltedpath, f"regrdir_{saltedname}", f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(reg))}.npy"
    ))
    
    print("load time:", (time.time()-loadstart))
    
    return [lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals]

if __name__ == "__main__":
    build()
