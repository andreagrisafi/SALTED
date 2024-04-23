import os
import sys
import time
import os.path as osp

import h5py
import numpy as np
from scipy import special

from salted import basis
from salted.sys_utils import ParseConfig

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

        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                avals = np.loadtxt(osp.join(
                    saltedpath, "basis", f"{spe}-{inp.qm.dfbasis}-alphas-L{l}.dat"
                ))
                if nmax[(spe,l)]==1:
                    alphas[(spe,l,0)] = float(avals)
                    sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
                else:
                    for n in range(nmax[(spe,l)]):
                        alphas[(spe,l,n)] = avals[n]
                        sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

        # compute integrals of basis functions (needed to a posteriori correction of the charge)
        for spe in species:
            for l in range(lmax[spe]+1):
                charge_integrals_temp = np.zeros(nmax[(spe,l)])
                for n in range(nmax[(spe,l)]):
                    inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                    charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                    charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
    
    loadstart = time.time()
   
    # Load feature space sparsification information if required 
    if sparsify:
        vfps = {}
        for lam in range(lmax_max+1):
            vfps[lam] = np.load(osp.join(
                saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    # Load training feature vectors and RKHS projection matrix
    Vmat = {}
    Mspe = {}
    power_env_sparse = {}
    for spe in species:
        for lam in range(lmax[spe]+1):
             # load RKHS projectors
             Vmat[(lam,spe)] = np.load(osp.join(
                 saltedpath,
                 f"equirepr_{saltedname}",
                 f"spe{spe}_l{lam}",
                 f"projector_M{Menv}_zeta{zeta}.npy",
             ))
             # load sparse equivariant descriptors
             power_env_sparse[(lam,spe)] = h5py.File(osp.join(
                 saltedpath,
                 f"equirepr_{saltedname}",
                 f"spe{spe}_l{lam}",
                 f"FEAT_M-{Menv}.h5"
             ), 'r')['sparse_descriptor'][:]
             if lam == 0:
                 Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
             # precompute projection on RKHS if linear model
             if zeta==1:
                 power_env_sparse[(lam,spe)] = np.dot(
                     Vmat[(lam,spe)].T, power_env_sparse[(lam,spe)]
                 )
 
    # load regression weights
    ntrain = int(inp.gpr.Ntrain*inp.gpr.trainfrac)
    if inp.system.field:
        weights = np.load(osp.join(
            saltedpath, f"regrdir_{saltedname}_field", f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(reg))}.npy"
        ))
    else:
        weights = np.load(osp.join(
            saltedpath, f"regrdir_{saltedname}", f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(reg))}.npy"
        ))
    
    print("load time:", (time.time()-loadstart))
    
    return [lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals]

if __name__ == "__main__":
    build()
