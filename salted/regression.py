import os
import sys
import time
import os.path as osp

import numpy as np


def build():

    sys.path.insert(0, './')
    import inp
    
    # sparse-GPR parameters
    M = inp.Menv
    reg = inp.regul
    zeta = inp.z
    
    if inp.field:
        kdir = f"kernels_{inp.saltedname}_field"
        fdir = f"rkhs-vectors_{inp.saltedname}_field"
        rdir = f"regrdir_{inp.saltedname}_field"
    else:
        kdir = f"kernels_{inp.saltedname}"
        fdir = f"rkhs-vectors_{inp.saltedname}"
        rdir = f"regrdir_{inp.saltedname}"
    
    # define training set size 
    ntrain = round(inp.trainfrac*inp.Ntrain)
    
    # load regression matrices
    Avec = np.load(osp.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Avec_N{ntrain}.npy"))
    totsize = Avec.shape[0]
    print("problem dimensionality:", totsize,flush=True)
    if totsize > 70000:
        raise ValueError(f"problem dimension too large ({totsize=}), minimize directly loss-function instead!")
    Bmat = np.load(osp.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Bmat_N{ntrain}.npy"))
    
    start = time.time()
    
    w = np.linalg.solve(Bmat+np.eye(totsize)*reg,Avec)
    
    print(f"regression time: {((time.time()-start)/60):.3f} minutes")
    
    np.save(osp.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(reg))}.npy"), w)

    return

if __name__ == "__main__":
    build()
