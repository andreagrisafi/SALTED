import os
import sys
import numpy as np

def build():

    sys.path.insert(0, './')
    import inp
    
    # dataset slice boundaries 
    blocksize = inp.blocksize
    
    if inp.field:
        rdir = f"regrdir_{inp.saltedname}_field"
    else:
        rdir = f"regrdir_{inp.saltedname}"
    
    # sparse-GPR parameters
    M = inp.Menv
    zeta = inp.z
    
    if inp.Ntrain%blocksize != 0:
        print("Please choose a blocksize which is an exact divisor of ntrain")
        return

    # load training set
    trainrangetot = np.loadtxt(os.path.join(inp.saltedpath, rdir, f"training_set_N{inp.Ntrain}.txt"), int)
    ntrain = round(inp.trainfrac*inp.Ntrain)
    
    nblocks = int(ntrain/blocksize)
    print("blocksize =",blocksize)
    print("number of blocks = ",nblocks)
    
    Avec = np.load(os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Avec_N{blocksize}_chunk0.npy"))
    Bmat = np.load(os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Bmat_N{blocksize}_chunk0.npy"))
    for iblock in range(1,nblocks):
        Avec += np.load(os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Avec_N{blocksize}_chunk{iblock}.npy"))
        Bmat += np.load(os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Bmat_N{blocksize}_chunk{iblock}.npy"))
    
    np.save(os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Avec_N{ntrain}.npy", Avec))
    np.save(os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"Bmat_N{ntrain}.npy", Bmat))
 
    return

if __name__ == "__main__":
    build()
