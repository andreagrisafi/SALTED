import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle
from scipy import sparse
import argparse

from salted import basis
from salted.sys_utils import read_system,get_atom_idx

def build():

    def add_command_line_arguments_contraction(parsetext):
        parser = argparse.ArgumentParser(description=parsetext)
        parser.add_argument("-j1", "--istart", type=int, default=0, help="starting index")
        parser.add_argument("-j2", "--iend",   type=int, default=0, help="ending index")
        args = parser.parse_args()
        return args
    
    args = add_command_line_arguments_contraction("dataset subselection")
    # dataset slice boundaries 
    iend = args.iend
    
    if iend==0:
        istart = 0
    else:
        istart = args.istart-1
    
    sys.path.insert(0, './')
    import inp
    
    if inp.field:
        fdir = "rkhs-vectors_"+inp.saltedname+"_field"
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        fdir = "rkhs-vectors_"+inp.saltedname
        rdir = "regrdir_"+inp.saltedname
    
    # system definition
    spelist = inp.species
    xyzfile = read(inp.filename,":")
    ndata = len(xyzfile)
    
    # basis definition
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    
    llist = []
    nlist = []
    for spe in spelist:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    llmax = max(llist)
    nnmax = max(nlist)
    
    # sparse-GPR parameters
    M = inp.Menv
    eigcut = inp.eigcut
    reg = inp.regul
    zeta = inp.z
    
    coefdir = inp.coefdir
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    
    p = sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf0.npz")
    totsize = p.shape[-1]
    print("problem dimensionality:", totsize,flush=True)
    if totsize>70000:
        print("ERROR: problem dimension too large, minimize directly loss-function instead!")
        sys.exit(0)
    
    if inp.average:
        # load average density coefficients
        av_coefs = {}
        for spe in spelist:
            av_coefs[spe] = np.load("averages_"+str(spe)+".npy")
    
    dirpath = os.path.join(inp.saltedpath, rdir)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    dirpath = os.path.join(inp.saltedpath+rdir+"/", "M"+str(M)+"_zeta"+str(zeta))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    
    # define training set at random
    dataset = list(range(ndata))
    #random.Random(3).shuffle(dataset)
    trainrangetot = dataset[:inp.Ntrain]
    np.savetxt(inp.saltedpath+rdir+"/training_set_N"+str(inp.Ntrain)+".txt",trainrangetot,fmt='%i')
    ntraintot = int(inp.trainfrac*inp.Ntrain)
    trainrange = trainrangetot[:ntraintot]
    ntrain = len(trainrange)
    
    print("computing regression matrices...")
    
    if iend==0:
        ilast = ndata
    else:
        ilast = iend
    
    Avec = np.zeros(totsize)
    Bmat = np.zeros((totsize,totsize))
    for iconf in trainrange[istart:ilast]:
        print("conf:", iconf+1,flush=True)
       
        start = time.time()
        # load reference QM data
        ref_coefs = np.load(inp.saltedpath+coefdir+"coefficients_conf"+str(iconf)+".npy")
        over = np.load(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy")
        psivec = sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npz")
        psi = psivec.toarray()
    
        if inp.average:
    
            # fill array of average spherical components
            Av_coeffs = np.zeros(ref_coefs.shape[0])
            i = 0
            for iat in range(natoms[iconf]):
                spe = atomic_symbols[iconf][iat]
                if spe in species:
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                               Av_coeffs[i] = av_coefs[spe][n]
                            i += 2*l+1
            
            # subtract average
            ref_coefs -= Av_coeffs
        
        ref_projs = np.dot(over,ref_coefs)
        
        Avec += np.dot(psi.T,ref_projs)
        Bmat += np.dot(psi.T,np.dot(over,psi))
    
        print("conf time =", time.time()-start)
    #    if iconf+1==50 or iconf+1==100 or iconf+1==200 or iconf+1==400: 
    #        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(iconf+1)+".npy",Avec/float(iconf+1))
    #        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(iconf+1)+".npy",Bmat/float(iconf+1))
    
    Avec /= float(ntrain)
    Bmat /= float(ntrain)
    
    if iend==0:
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+".npy",Avec)
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+".npy",Bmat)
    else:
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+"_chunck"+str(istart+1)+"-"+str(iend)+".npy",Avec)
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+"_chunck"+str(istart+1)+"-"+str(iend)+".npy",Bmat)

    return

if __name__ == "__main__":
    build()
