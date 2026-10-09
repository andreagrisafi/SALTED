import os
import os.path as osp
import random
import time

import numpy as np
from scipy import sparse

from salted import get_averages
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx,
    load_overlap,
    read_system,
)


def build():

    build_start_time = time.time()
    inp = ParseConfig().parse_input()

    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath
    comm, size, rank, parallel = detect_mpi()

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()

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
        if rank==0:
            get_averages.build()
        if parallel:
            comm.Barrier()
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
    ntrain = round(inp.gpr.trainfrac*inp.gpr.Ntrain)
    trainrange = sorted(trainrangetot[:ntrain])

    """
    Calculate regression matrices in parallel or serial mode.
    """

    if parallel:
        """ check partitioning """
        assert size > 1, "Please run in serial mode if using a single MPI task"
        check_MPI_tasks_count(comm, ntrain, "training structures")
        this_task_trainrange = distribute_jobs(comm, trainrange)
        """ calculate and gather """
        if inp.salted.verbose:
            print(f"Task {rank} handling structures: {format_index_ranges(this_task_trainrange,True)}", flush=True)
        [Avec, Bmat, loop_start_time] = matrices(this_task_trainrange, ntrain,av_coefs,rank)
        matrices_end_time = time.time()
        comm.Barrier()
        barrier_end_time = time.time()
        """ sum the matrices of all tasks into task 0 """
        from mpi4py import MPI
        rows_per_block = max(1, 2**26 // Bmat.shape[1]) # Max 2**26 entries (512 MB of doubles) per block
        blocks = [Avec] + [Bmat[i:i+rows_per_block] for i in range(0, Bmat.shape[0], rows_per_block)]
        for block in blocks:
            if rank == 0:
                comm.Reduce(MPI.IN_PLACE, block, op=MPI.SUM, root=0) # sum the contributions from all tasks into task 0
            else:
                comm.Reduce(block, None, op=MPI.SUM, root=0) # send the contributions from this task to task 0
        reduce_end_time = time.time()
    else:
        print("Running in serial mode")
        [Avec, Bmat, loop_start_time] = matrices(trainrange,ntrain,av_coefs,rank)
        matrices_end_time = time.time()
        barrier_end_time = reduce_end_time = matrices_end_time

    if rank==0:
        np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Avec_N{ntrain}.npy"), Avec)
        np.save(osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"Bmat_N{ntrain}.npy"), Bmat)
    save_end_time = time.time()

    if inp.salted.verbose:
        print(
            f"Task {rank}, time outside the configuration loop: "
            f"setup = {(loop_start_time - build_start_time):.2f} s, "
            f"wait for other tasks = {(barrier_end_time - matrices_end_time):.2f} s, "
            f"reduction over tasks = {(reduce_end_time - barrier_end_time):.2f} s, "
            f"save = {(save_end_time - reduce_end_time):.2f} s",
            flush=True,
        )


def _compute_sparse_operations(psivec, ref_projs, over, sparse_algorithm, layout=None, symbols=None, bmat=None):
    """
    Compute sparse matrix operations with fallback logic.

    Args:
        psivec: Sparse matrix (scipy.sparse)
        ref_projs: Dense vector/matrix
        over: Dense overlap matrix
        sparse_algorithm: "numba", "dense", "omp_sparse" or "dense_blocks"
        layout: dense_blocks.BlockLayout (dense_blocks only)
        symbols: atomic symbols of the configuration (dense_blocks only)
        bmat: regression matrix

    Returns:
        (avec_contrib, bmat_contrib, algorithm_used, timings)
        bmat_contrib is None if the contribution has already been added to bmat (dense_blocks case)
        timings: wall times of step 1 ("step1", over @ psi) and step 2 ("step2", psi.T @ (over @ psi))
        in seconds, reported by dense_blocks only (empty for the other algorithms)
    """
    if sparse_algorithm == "dense_blocks":
        from salted.dense_blocks import LayoutMismatch, hessian_contribution
        try:
            timings = hessian_contribution(over, psivec, layout, symbols, bmat)
            avec_contrib = psivec.T @ ref_projs               # O(nnz) vector multiply
            return avec_contrib, None, "dense_blocks", timings
        except LayoutMismatch as e:
            print(f"Warning: psi does not have the layout dense_blocks expects ({e}), falling back to numba", flush=True)
            sparse_algorithm = "numba"

    if sparse_algorithm == "omp_sparse":
        try:
            from salted.omp_sparse import dense_dot_sparse, sparse_transpose_dot_dense

            # Avec += psi.T @ ref_projs
            avec_contrib = sparse_transpose_dot_dense(psivec, ref_projs)

            # Bmat += psi.T @ (over @ psi)
            bmat_contrib = sparse_transpose_dot_dense(psivec, dense_dot_sparse(over, psivec))

            return avec_contrib, bmat_contrib, "omp_sparse", {}

        except Exception as e:
            print(f"Warning: omp_sparse unavailable ({e}), falling back to numba", flush=True)
            sparse_algorithm = "numba"

    if sparse_algorithm == "numba":
        from salted.numba_sparse import get_hessian_engine
        N_df, K_rkhs = psivec.shape
        avec_contrib = psivec.T @ ref_projs               # O(nnz) vector multiply
        engine       = get_hessian_engine(N_df, K_rkhs)
        bmat_contrib = engine.compute(over, psivec)
        return avec_contrib, bmat_contrib, "numba", {}

    # Dense fallback (original behavior)
    psi_dense = psivec.toarray()
    avec_contrib = np.dot(psi_dense.T, ref_projs)
    bmat_contrib = np.dot(psi_dense.T, np.dot(over, psi_dense))

    return avec_contrib, bmat_contrib, "dense", {}


def _add_times(total, times):
    for key, value in times.items():
        total[key] = total.get(key, 0.0) + value


def _format_conf_details(iconf, nnz, nentries, flops, times):
    """Return one log line with nnz(psi) of one configuration and, if the algorithm reports
    them (dense_blocks), the times of steps 1 and 2 and the speed of their 2 nnz (N_df + K)
    operations. Kept for comparisons with minimize_loss; to be removed afterwards.
    """
    line = f"conf {iconf} details: nnz(psi) = {nnz} ({100.0 * nnz / nentries:.3f}% of entries)"
    if "step1" in times:
        t1, t2 = times["step1"], times["step2"]
        rate = flops / (t1 + t2) / 1e9 if t1 + t2 > 0 else 0.0
        line += f", step 1 = {t1:.2f} s, step 2 = {t2:.2f} s, {rate:.1f} GFLOP/s ({flops / 1e9:.1f} GFLOP)"
    return line


def matrices(trainrange,ntrain,av_coefs,rank):

    inp = ParseConfig().parse_input()

    saltedname, saltedpath = inp.salted.saltedname, inp.salted.saltedpath
    # sparse-GPR parameters
    Menv = inp.gpr.Menv
    zeta = inp.gpr.z
    fdir = f"rkhs-vectors_{saltedname}"
    sparse_algorithm = inp.gpr.sparse_algorithm

    if rank == 0:
        print(f"Using sparse algorithm: {sparse_algorithm}", flush=True)

    if inp.salted.saltedtype=="density-response":
        p = sparse.load_npz(osp.join(
            saltedpath, fdir, f"M{Menv}_zeta{zeta}", "psi-nm_conf0_x.npz"
        ))
    else:
        p = sparse.load_npz(osp.join(
            saltedpath, fdir, f"M{Menv}_zeta{zeta}", "psi-nm_conf0.npz"
        ))

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

    totsize = p.shape[-1]
    if rank == 0: print("problem dimensionality:", totsize,flush=True)
    if totsize>100000:
        raise ValueError(f"problem dimension too large ({totsize=}), minimize directly loss-function instead!")

    layout = None
    if sparse_algorithm == "dense_blocks":
        from salted.dense_blocks import BlockLayout
        projector = "projector-response" if inp.salted.saltedtype == "density-response" else "projector"
        layout = BlockLayout.from_projectors(osp.join(
            saltedpath, f"equirepr_{saltedname}", f"{projector}_M{Menv}_zeta{zeta}.h5"
        ), species, lmax, nmax)
        if layout.ncols != totsize:
            if rank == 0: print(f"Warning: the RKHS projectors give {layout.ncols} columns of psi, the psi files have {totsize}; falling back to numba", flush=True)
            sparse_algorithm, layout = "numba", None

    if rank == 0: print("computing regression matrices...")

    Avec = np.zeros(totsize)
    Bmat = np.zeros((totsize,totsize))
    total_start_time = time.time()
    total_io_time, total_compute_time = 0.0, 0.0
    
    for iconf in trainrange: # loop over training configurations

        start_time = time.time()
        io_time, compute_time = 0.0, 0.0
        times = {} # times of steps 1 and 2 for this configuration, if the algorithm reports them
        nnz, nentries, flops = 0, 0, 0.0

        if inp.salted.saltedtype=="density":

            # load reference QM data
            t0 = time.time()
            ref_coefs = np.load(osp.join(
                saltedpath, "coefficients", f"coefficients_conf{iconf}.npy" # Load reference coefficients c for the current configuration
            ))
            over = load_overlap(osp.join(
                saltedpath, "overlaps", f"overlap_conf{iconf}.npy" # Load overlap matrix S for the current configuration
            ))
            psivec = sparse.load_npz(osp.join(
                saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz" # Load sparse RKHS vectors for the current configuration
            ))
            t1 = time.time()
            io_time += t1 - t0

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

            ref_projs = np.dot(over,ref_coefs) # S*c

            # Use sparse operations with automatic fallback
            avec_contrib, bmat_contrib, algorithm_used, kernel_times = _compute_sparse_operations(
                psivec, ref_projs, over, sparse_algorithm, layout, atomic_symbols[iconf], Bmat
            )
            _add_times(times, kernel_times)
            if algorithm_used != sparse_algorithm:
                # set sparse_algorithm to algorithm_used
                print(f"Warning: Using fallback dense algorithm for conf {iconf} due to failure in {sparse_algorithm}, set current sparse_algorithm to {algorithm_used}", flush=True)
                sparse_algorithm = algorithm_used
            Avec += avec_contrib # A = Psi^T @ S @ c
            if bmat_contrib is not None: # for cases other than dense_blocks
                Bmat += bmat_contrib # B = Psi^T @ S @ Psi
            nnz += psivec.nnz # count non-zero entries in psivec
            nentries += psivec.shape[0] * psivec.shape[1] # count total entries in psivec
            flops += 2.0 * psivec.nnz * (psivec.shape[0] + psivec.shape[1])
            compute_time += time.time() - t1


        elif inp.salted.saltedtype=="density-response":

            t0 = time.time()
            over = load_overlap(osp.join(
                saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
            ))
            io_time += time.time() - t0

            for icart in ["x","y","z"]:

                t0 = time.time()
                ref_coefs = np.load(osp.join(
                    saltedpath, "coefficients", f"{icart}/coefficients_conf{iconf}.npy"
                ))
                psivec = sparse.load_npz(osp.join(
                    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                ))
                t1 = time.time()
                io_time += t1 - t0

                ref_projs = np.dot(over,ref_coefs)

                # Use sparse operations with automatic fallback
                avec_contrib, bmat_contrib, algorithm_used, kernel_times = _compute_sparse_operations(
                    psivec, ref_projs, over, sparse_algorithm, layout, atomic_symbols[iconf], Bmat
                )
                _add_times(times, kernel_times)
                if algorithm_used != sparse_algorithm:
                    print(f"Warning: Using fallback algorithm for conf {iconf} ({icart}) due to failure in {sparse_algorithm}, set current sparse_algorithm to {algorithm_used}", flush=True)
                    sparse_algorithm = algorithm_used
                Avec += avec_contrib
                if bmat_contrib is not None: # for cases other than dense_blocks
                    Bmat += bmat_contrib
                nnz += psivec.nnz # count non-zero entries in psivec
                nentries += psivec.shape[0] * psivec.shape[1] # count total entries in psivec
                flops += 2.0 * psivec.nnz * (psivec.shape[0] + psivec.shape[1]) # count flops for this cartesian component
                compute_time += time.time() - t1

        del over, psivec

        total_io_time += io_time
        total_compute_time += compute_time
        if inp.salted.verbose:
            print(f"conf {iconf}, time = {(time.time() - start_time):.2f} s (I/O = {io_time:.2f} s, compute = {compute_time:.2f} s)", flush=True)
            print(_format_conf_details(iconf, nnz, nentries, flops, times), flush=True)

    if inp.salted.verbose: print(f"Task {rank}, total time for {len(trainrange)} structures = {(time.time() - total_start_time):.2f} s (I/O = {total_io_time:.2f} s, compute = {total_compute_time:.2f} s)", flush=True)

    Avec /= float(ntrain)
    Bmat /= float(ntrain)

    return [Avec,Bmat,total_start_time]

if __name__ == "__main__":
    build()
