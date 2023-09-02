import os
import sys
import numpy as np

def build():

    sys.path.insert(0, './')
    import inp
    
    # dataset slice boundaries 
    blocksize = inp.blocksize
    
    if inp.field:
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        rdir = "regrdir_"+inp.saltedname
    
    # sparse-GPR parameters
    M = inp.Menv
    zeta = inp.z
    
    # load training set 
    ntrain = int(inp.trainfrac*inp.Ntrain)
    
    if ntrain%blocksize != 0:
        print("Please choose a blocksize which is an exact divisor of ntrain")
        return

    nblocks = int(ntrain/blocksize)
    print("number of blocks = ",nblocks)
    
    Avec = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(blocksize)+"_chunk0.npy")
    Bmat = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(blocksize)+"_chunk0.npy")
    for iblock in range(1,nblocks):
        Avec += np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(blocksize)+"_chunk"+str(iblock)+".npy")
        Bmat += np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(blocksize)+"_chunk"+str(iblock)+".npy")
    
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+".npy",Avec)
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+".npy",Bmat)
 
    return

if __name__ == "__main__":
    build()
