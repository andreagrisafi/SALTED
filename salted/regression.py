import os
import sys
import numpy as np
import time
from scipy import sparse

from salted import basis

def build():

    sys.path.insert(0, './')
    import inp
    
    # sparse-GPR parameters
    M = inp.Menv
    reg = inp.regul
    zeta = inp.z
    
    if inp.field:
        kdir = "kernels_"+inp.saltedname+"_field"
        fdir = "rkhs-vectors_"+inp.saltedname+"_field"
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        kdir = "kernels_"+inp.saltedname
        fdir = "rkhs-vectors_"+inp.saltedname
        rdir = "regrdir_"+inp.saltedname
    
    # basis definition
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    
    llist = []
    nlist = []
    for spe in inp.species:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    llmax = max(llist)
    nnmax = max(nlist)
    
    # compute the weight-vector size 
    totsize = 0
    iii=0
    for spe in inp.species:
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = np.load(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(0)+".npy").shape[1]
                totsize += Mcut
                iii+=1
    
    print("problem dimensionality:", totsize,flush=True)
    if totsize>70000:
        print("ERROR: problem dimension too large, minimize directly loss-function instead!")
        sys.exit(0)
    
    # define training set size 
    ntrain = int(inp.trainfrac*inp.Ntrain)
    
    # load regression matrices
    Avec = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+".npy")
    Bmat = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+".npy")
    
    start = time.time()
    
    w = np.linalg.solve(Bmat+np.eye(totsize)*reg,Avec)
    
    print("regression time:", (time.time()-start)/60, "minutes")
    
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",w)

    return

if __name__ == "__main__":
    build()
