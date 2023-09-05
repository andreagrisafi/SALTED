import os
import sys
import numpy as np
import time
import random
from scipy import sparse

from salted.sys_utils import read_system,get_atom_idx,get_conf_range

def build():
    
    sys.path.insert(0, './')
    import inp
    
    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
        size = 1
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    if inp.field:
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        rdir = "regrdir_"+inp.saltedname
    
    # sparse-GPR parameters
    M = inp.Menv
    zeta = inp.z
    
    if rank == 0:    
        dirpath = os.path.join(inp.saltedpath, rdir)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        dirpath = os.path.join(inp.saltedpath+rdir+"/", "M"+str(M)+"_zeta"+str(zeta))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    # define training set at random
    dataset = list(range(ndata))
    random.Random(3).shuffle(dataset)
    trainrangetot = dataset[:inp.Ntrain]
    np.savetxt(inp.saltedpath+rdir+"/training_set_N"+str(inp.Ntrain)+".txt",trainrangetot,fmt='%i')
    ntraintot = int(inp.trainfrac*inp.Ntrain)
    trainrange = trainrangetot[:ntraintot]
    ntrain = len(trainrange)

    try:
        blocksize = inp.blocksize
        if blocksize > 0:
            blocks = True
        else:
            blocks = False
    except:
        blocksize = ntrain
        blocks = False

    if not blocks and size > 1:
        print("Please run serially if computing a single matrix, or add inp.blocksize>0 to the input file to compute the matrix blockwise and in parallel.")
        return

    if blocks:
        if ntrain%blocksize != 0:
            print("Please choose a blocksize which is an exact divisor of ntrain")
            return
        nblocks = int(ntrain/blocksize)
        j = 0
        for i in range(nblocks):
            if rank==(i-j*size): matrices(i,trainrange[i*blocksize:(i+1)*blocksize],rank)
#            print(rank,i,i+1,(j+1)*size,j)
            if i+1 == (j+1)*size: j += 1

    else:
        matrices(-1,trainrange,rank)

def matrices(block_idx,trainrange,rank):
    
    sys.path.insert(0, './')
    import inp
    print("Task",rank,"handling structures:",trainrange)

    if inp.field:
        fdir = "rkhs-vectors_"+inp.saltedname+"_field"
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        fdir = "rkhs-vectors_"+inp.saltedname
        rdir = "regrdir_"+inp.saltedname
    
    # sparse-GPR parameters
    M = inp.Menv
    zeta = inp.z
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    p = sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf0.npz")
    totsize = p.shape[-1]
    if rank == 0: print("problem dimensionality:", totsize,flush=True)
    if totsize>70000:
        print("ERROR: problem dimension too large, minimize directly loss-function instead!")
        sys.exit(0)
    
    if inp.average:
        # load average density coefficients
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load("averages_"+str(spe)+".npy")
   
    if rank == 0: print("computing regression matrices...")
    
    Avec = np.zeros(totsize)
    Bmat = np.zeros((totsize,totsize))
    for iconf in trainrange:
        print("conf:", iconf+1,flush=True)
       
        start = time.time()
        # load reference QM data
        ref_coefs = np.load(inp.saltedpath+"coefficients_"+inp.saltedname+"/coefficients_conf"+str(iconf)+".npy")
        over = np.load(inp.saltedpath+"overlaps_"+inp.saltedname+"/overlap_conf"+str(iconf)+".npy")
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
   
    ntrain = len(trainrange)
    Avec /= float(ntrain)
    Bmat /= float(ntrain)
    
    if block_idx == -1:
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+".npy",Avec)
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+".npy",Bmat)
    else:
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+"_chunk"+str(block_idx)+".npy",Avec)
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+"_chunk"+str(block_idx)+".npy",Bmat)

    return

if __name__ == "__main__":
    build()
