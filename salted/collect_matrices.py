import os
import sys
import numpy as np

from salted.sys_utils import ParseConfig


def build():

    inp = ParseConfig().parse_input()
    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath

    # dataset slice boundaries 
    blocksize = inp.gpr.blocksize
    
    if inp.system.field:
        rdir = f"regrdir_{saltedname}_field"
    else:
        rdir = f"regrdir_{saltedname}"
    
    # sparse-GPR parameters
    Menv, zeta = inp.gpr.Menv, inp.gpr.z
    
    if inp.gpr.Ntrain % blocksize != 0:
        print("Please choose a blocksize which is an exact divisor of ntrain")
        return

    # load training set
    trainrangetot = np.loadtxt(os.path.join(saltedpath, rdir, f"training_set_N{inp.gpr.Ntrain}.txt"), int)
    ntrain = round(inp.gpr.trainfrac * inp.gpr.Ntrain)
    
    nblocks = int(ntrain/blocksize)
    print("blocksize =",blocksize)
    print("number of blocks = ",nblocks)
    
    Avec = np.load(os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{blocksize}_chunk0.npy"))
    Bmat = np.load(os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{blocksize}_chunk0.npy"))
    for iblock in range(1,nblocks):
        Avec += np.load(os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{blocksize}_chunk{iblock}.npy"))
        Bmat += np.load(os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{blocksize}_chunk{iblock}.npy"))
    
    np.save(os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{ntrain}.npy"), Avec)
    np.save(os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{ntrain}.npy"), Bmat)
 
    return

if __name__ == "__main__":
    build()
