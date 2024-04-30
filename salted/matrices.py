import os
import sys
import time
import random
import os.path as osp

import numpy as np
from scipy import sparse

from salted import get_averages
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

def build():

    inp = ParseConfig().parse_input()

    parallel = inp.system.parallel
    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath

    if parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        comm = None
        rank = 0
        size = 1

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

    if inp.system.field:
        rdir = f"regrdir_{saltedname}_field"
    else:
        rdir = f"regrdir_{saltedname}"
    
    # sparse-GPR parameters
    Menv = inp.gpr.Menv
    zeta = inp.gpr.z
    
    if rank == 0:    
        dirpath = os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    av_coefs = {} # keep outside logical
    if inp.system.average:
        # compute average density coefficients
        if rank==0: get_averages.build()
        if parallel: comm.Barrier()
        # load average density coefficients
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    if size > 1: comm.Barrier()

    # define training set at random or sequentially
    dataset = list(range(ndata))
    if inp.gpr.trainsel=="sequential":
        trainrangetot = dataset[:inp.gpr.Ntrain]
    elif inp.gpr.trainsel=="random":
        random.Random(3).shuffle(dataset)
        trainrangetot = dataset[:inp.gpr.Ntrain]
    else:
        raise ValueError(f"training set selection {inp.gpr.trainsel=} not available!")
    np.savetxt(osp.join(
        saltedpath, rdir, f"training_set_N{inp.gpr.Ntrain}.txt"
    ), trainrangetot, fmt='%i')
    ntrain = int(inp.gpr.trainfrac*inp.gpr.Ntrain)
    trainrange = trainrangetot[:ntrain]

    if inp.gpr.blocksize==0:
        blocksize = ntrain 
        blocks = False 
    else:
        if parallel==False:
            print("Please activate parallel mode when using inp.gpr.blocksize to compute matrices in blocks!")
            return
        blocksize = inp.gpr.blocksize
        blocks = True

    if not blocks and size > 1:
        print("Please run serially if computing a single matrix, or add inp.gpr.blocksize>0 to the input file to compute the matrix blockwise and in parallel!")
        return
    
    if parallel: print("Task",rank,"handling structures:",trainrange[rank*blocksize:(rank+1)*blocksize])

    if blocks:

        if ntrain%blocksize != 0:
            print("Please choose a blocksize which is an exact divisor of inp.gpr.Ntrain*inp.gpr.trainfrac!")
            return
        nblocks = int(ntrain/blocksize)
        if nblocks != size:
            print(f"Please choose a number of MPI tasks consistent with the number of blocks {nblocks}!")
            return
        regr_matr = matrices(trainrange[rank*blocksize:(rank+1)*blocksize],ntrain,av_coefs,rank)

    else:

        regr_matr = matrices(trainrange,ntrain,av_coefs,rank)   
 
    if parallel:
        comm.Barrier()
        Avec = comm.allreduce(regr_matr[0])
        Bmat = comm.allreduce(regr_matr[1])
    else:
        Avec = regr_matr[0]
        Bmat = regr_matr[1]

    if rank==0: 
        np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{ntrain}.npy"), Avec)
        np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{ntrain}.npy"), Bmat)


def matrices(trainrange,ntrain,av_coefs,rank):
    
    inp = ParseConfig().parse_input()

    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath

    if inp.system.field:
        fdir = f"rkhs-vectors_{saltedname}_field"
        rdir = f"regrdir_{saltedname}_field"
    else:
        fdir = f"rkhs-vectors_{saltedname}"
        rdir = f"regrdir_{saltedname}"
    
    # sparse-GPR parameters
    Menv = inp.gpr.Menv
    zeta = inp.gpr.z
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    p = sparse.load_npz(osp.join(
        saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf0.npz"
    ))
    totsize = p.shape[-1]
    if rank == 0: print("problem dimensionality:", totsize,flush=True)
    if totsize>70000:
        raise ValueError(f"problem dimension too large ({totsize=}), minimize directly loss-function instead!")
    
    if rank == 0: print("computing regression matrices...")
    
    Avec = np.zeros(totsize)
    Bmat = np.zeros((totsize,totsize))
    for iconf in trainrange:
        print("conf:", iconf+1,flush=True)
       
        start = time.time()
        # load reference QM data
        ref_coefs = np.load(osp.join(
            saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"
        ))
        over = np.load(osp.join(
            saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
        ))
        psivec = sparse.load_npz(osp.join(
            saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
        ))
        psi = psivec.toarray()
    
        if inp.system.average:
    
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
   
    Avec /= float(ntrain)
    Bmat /= float(ntrain)
    
    return [Avec,Bmat]

if __name__ == "__main__":
    build()
