import os
import os.path as osp
import sys
import time

import numpy as np

from salted.sys_utils import ParseConfig


def build():

    inp = ParseConfig().parse_input()
    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath

    # sparse-GPR parameters
    Menv = inp.gpr.Menv
    regul = inp.gpr.regul
    zeta = inp.gpr.z

    fdir = f"rkhs-vectors_{saltedname}"
    rdir = f"regrdir_{saltedname}"

    # define training set size
    ntrain = round(inp.gpr.trainfrac*inp.gpr.Ntrain)

    # load regression matrices
    Avec = np.load(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{ntrain}.npy"))
    totsize = Avec.shape[0]
    print("problem dimensionality:", totsize,flush=True)
    if totsize > 100000:
        raise ValueError(f"problem dimension too large ({totsize=}), minimize directly loss-function instead!")
    Bmat = np.load(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{ntrain}.npy"))

    start = time.time()

    w = np.linalg.solve(Bmat+np.eye(totsize)*regul,Avec)

    print(f"regression time: {((time.time()-start)/60):.3f} minutes",flush=True)

    np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(regul))}.npy"), w)

    return

if __name__ == "__main__":
    build()
