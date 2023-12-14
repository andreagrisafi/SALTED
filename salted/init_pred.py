import os
import sys
import time
import h5py
import numpy as np
from scipy import special

from salted import basis

def build():

    sys.path.insert(0, './')
    import inp

    saltedname = inp.saltedname
    species = inp.species 
    ncut = inp.ncut 
    M = inp.Menv
    zeta = inp.z
    reg = inp.regul
  
    # read basis
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    llist = []
    nlist = []
    for spe in species:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    lmax_max = max(llist)

    if inp.qmcode=="cp2k":

        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                avals = np.loadtxt(inp.saltedpath+"basis/"+spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat")
                if nmax[(spe,l)]==1:
                    alphas[(spe,l,0)] = float(avals)
                    sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
                else:
                    for n in range(nmax[(spe,l)]):
                        alphas[(spe,l,n)] = avals[n]
                        sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

        # compute integrals of basis functions (needed to a posteriori correction of the charge)
        charge_integrals = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                charge_integrals_temp = np.zeros(nmax[(spe,l)])
                for n in range(nmax[(spe,l)]):
                    inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                    charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                    charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
    
    loadstart = time.time()
    
    # Load training feature vectors and RKHS projection matrix 
    Mspe = {}
    power_env_sparse = {}
    if inp.field: power_env_sparse_field = {}
    Vmat = {}
    vfps = {}
    if inp.field: vfps_field = {}
    for lam in range(lmax_max+1):
        # Load sparsification details
        if ncut > 0: vfps[lam] = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+".npy")
        if ncut > 0 and inp.field: vfps_field[lam] = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+"_field.npy")
        for spe in species:
            # load sparse equivariant descriptors 
            power_env_sparse[(lam,spe)] = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-M-"+str(M)+".h5",'r')[spe][:]
            if inp.field: power_env_sparse_field[(lam,spe)] = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-M-"+str(M)+"_field.h5",'r')[spe][:]
            if lam == 0: Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
            # load RKHS projectors 
            if inp.field:
                Vmat[(lam,spe)] = np.load(inp.saltedpath+"kernels_"+saltedname+"_field/spe"+str(spe)+"_l"+str(lam)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy")
            else:
                Vmat[(lam,spe)] = np.load(inp.saltedpath+"kernels_"+saltedname+"/spe"+str(spe)+"_l"+str(lam)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy")
            # precompute projection on RKHS if linear model 
            if zeta==1:
                power_env_sparse[(lam,spe)] = np.dot(Vmat[(lam,spe)].T,power_env_sparse[(lam,spe)])
                if inp.field: power_env_sparse_field[(lam,spe)] = np.dot(Vmat[(lam,spe)].T,power_env_sparse_field[(lam,spe)])
    
    # load regression weights
    ntrain = int(inp.Ntrain*inp.trainfrac)
    if inp.field:
        weights = np.load(inp.saltedpath+"regrdir_"+saltedname+"_field/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
    else:
        weights = np.load(inp.saltedpath+"regrdir_"+saltedname+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
    
    print("load time:", (time.time()-loadstart))
    
    return [lmax,nmax,lmax_max,weights,power_env_sparse,Vmat,vfps,charge_integrals]

if __name__ == "__main__":
    build()
