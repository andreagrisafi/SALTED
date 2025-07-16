import os
import os.path as osp
import random
import sys
import time

import numpy as np
from scipy import sparse

from salted import get_averages
from salted.sys_utils import ParseConfig, get_atom_idx, read_system

try:
    # raise ImportError  # for testing without omp_sparse
    from omp_sparse import multiply_dense_sparse
    OMP_SPARSE_AVAIL = True
except ImportError:
    OMP_SPARSE_AVAIL = False

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

    rdir = f"regrdir_{saltedname}"

    # sparse-GPR parameters
    Menv = inp.gpr.Menv
    zeta = inp.gpr.z

    if rank == 0:
        dirpath = os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    av_coefs = {} # keep outside logical
    if inp.system.average:
        # compute average density coefficients
        if rank==0: get_averages.build()
        if parallel: comm.Barrier()
        # load average density coefficients
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    if parallel:
        comm.Barrier()

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

    """
    Calculate regression matrices in parallel or serial mode.
    """

    if parallel:
        print("Running in parallel mode")
        """ check partitioning """
        assert size > 1, "Please run in serial mode if using a single MPI task"
        assert inp.gpr.blocksize > 0, "Please set inp.gpr.blocksize > 0 when running in parallel mode"
        assert isinstance(inp.gpr.blocksize, int), "Please set inp.gpr.blocksize as an integer"
        blocksize = inp.gpr.blocksize
        assert ntrain % blocksize == 0, \
            "Please choose a blocksize which is an exact divisor of inp.gpr.Ntrain * inp.gpr.trainfrac!"
        nblocks = int(ntrain/blocksize)
        assert nblocks == size, \
            f"Please choose a number of MPI tasks (current ntasks={size}) consistent with " \
            f"the number of blocks {nblocks} = inp.gpr.Ntrain * inp.gpr.trainfrac / inp.gpr.blocksize!"
        this_task_trainrange = trainrange[rank*blocksize:(rank+1)*blocksize]
        """ calculate and gather """
        print(f"Task {rank} handling structures: {this_task_trainrange}")
        [Avec, Bmat] = matrices(this_task_trainrange, ntrain,av_coefs,rank)
        comm.Barrier()
        """ reduce matrices in slices to avoid MPI overflows """
        nslices = int(np.ceil(len(Avec) / 100.0))
        for islice in range(nslices-1):
            Avec[islice*100:(islice+1)*100] = comm.allreduce(Avec[islice*100:(islice+1)*100])
            Bmat[islice*100:(islice+1)*100] = comm.allreduce(Bmat[islice*100:(islice+1)*100])
        Avec[(nslices-1)*100:] = comm.allreduce(Avec[(nslices-1)*100:])
        Bmat[(nslices-1)*100:] = comm.allreduce(Bmat[(nslices-1)*100:])
    else:
        print("Running in serial mode")
        assert inp.gpr.blocksize == 0, "Please DON'T provide inp.gpr.blocksize in inp.yaml when running in serial mode"
        [Avec, Bmat] = matrices(trainrange,ntrain,av_coefs,rank)

    if rank==0:
        np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{ntrain}.npy"), Avec)
        np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{ntrain}.npy"), Bmat)


def matrices(trainrange,ntrain,av_coefs,rank):

    inp = ParseConfig().parse_input()

    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath
    # sparse-GPR parameters
    Menv = inp.gpr.Menv
    zeta = inp.gpr.z
    fdir = f"rkhs-vectors_{saltedname}"

    if inp.salted.saltedtype=="density-response":
        p = sparse.load_npz(osp.join(
            saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf0_x.npz"
        ))
    else:
        p = sparse.load_npz(osp.join(
            saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf0.npz"
        ))

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

    totsize = p.shape[-1]
    if rank == 0: print("problem dimensionality:", totsize,flush=True)
    if totsize>100000:
        raise ValueError(f"problem dimension too large ({totsize=}), minimize directly loss-function instead!")

    if rank == 0:
        print("computing regression matrices...")
        start_time = time.time()

    Avec = np.zeros(totsize)
    Bmat = np.zeros((totsize,totsize))
    for iconf in trainrange:
        print("conf:", iconf+1,flush=True)

        start = time.time()

        if inp.salted.saltedtype=="density":

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
            if not OMP_SPARSE_AVAIL:
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

            if OMP_SPARSE_AVAIL:
                Avec += multiply_dense_sparse(ref_projs.reshape(1,-1), psivec).reshape(-1)
                Bmat += multiply_dense_sparse(multiply_dense_sparse(over, psivec).T, psivec).T
            else:
                Avec += np.dot(psi.T,ref_projs)
                Bmat += np.dot(psi.T,np.dot(over,psi))

        elif inp.salted.saltedtype=="density-response":

            over = np.load(osp.join(
                saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
            ))

            for icart in ["x","y","z"]:

                ref_coefs = np.load(osp.join(
                    saltedpath, "coefficients", f"{icart}/coefficients_conf{iconf}.npy"
                ))
                psivec = sparse.load_npz(osp.join(
                    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                ))
                psi = psivec.toarray()

                ref_projs = np.dot(over,ref_coefs)

                Avec += np.dot(psi.T,ref_projs)
                Bmat += np.dot(psi.T,np.dot(over,psi))

        print("conf time =", time.time()-start)

    Avec /= float(ntrain)
    Bmat /= float(ntrain)

    if rank == 0:
        print(f"total time = {time.time()-start_time:.3e} s", flush=True)

    return [Avec,Bmat]

if __name__ == "__main__":
    build()