import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse

from salted import sys_utils
from salted import basis

def build():

    def add_command_line_arguments_contraction(parsetext):
        parser = argparse.ArgumentParser(description=parsetext)
        parser.add_argument("-bs", "--blocksize", type=int, default=0, help="block size")
        args = parser.parse_args()
        return args
    
    args = add_command_line_arguments_contraction("dataset subselection")
    # dataset slice boundaries 
    blocksize = args.blocksize
    
    sys.path.insert(0, './')
    import inp
    
    if inp.field:
        kdir = "kernels_"+inp.saltedname+"_field"
        fdir = "rkhs-vectors_"+inp.saltedname+"_field"
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        kdir = "kernels_"+inp.saltedname
        fdir = "rkhs-vectors_"+inp.saltedname
        rdir = "regrdir_"+inp.saltedname
    
    # system definition
    species = inp.species
    xyzfile = read(inp.filename,":")
    ndata = len(xyzfile)
    
    # basis definition
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    
    llist = []
    nlist = []
    for spe in species:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    llmax = max(llist)
    nnmax = max(nlist)
    
    # sparse-GPR parameters
    M = inp.Menv
    reg = inp.regul
    zeta = inp.z
    
    # load training set 
    trainrangetot = np.loadtxt(inp.saltedpath+rdir+"/training_set_N"+str(inp.Ntrain)+".txt",int)
    ntrain = int(inp.trainfrac*inp.Ntrain)
    
    nblocks = int(ntrain/blocksize)
    print("number of blocks = ",nblocks)
    
    # compute the weight-vector size 
    totsize = 0
    for spe in species:
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = np.load(inp.saltedpath+kdir+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(0)+".npy").shape[1]
                totsize += Mcut
    
    Avec = np.zeros(totsize)
    Bmat = np.zeros((totsize,totsize))
    for iblock in range(nblocks):
        print("block", iblock+1)
        istart = iblock*blocksize+1 
        iend = (iblock+1)*blocksize
        Avec += np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(inp.Ntrain)+"_chunck"+str(istart)+"-"+str(iend)+".npy")
        Bmat += np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(inp.Ntrain)+"_chunck"+str(istart)+"-"+str(iend)+".npy")
    
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+".npy",Avec)
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+".npy",Bmat)
 
    return

if __name__ == "__main__":
    build()
