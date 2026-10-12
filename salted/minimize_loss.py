import os
import os.path as osp
import random
import sys
import time

import numba
import numpy as np
from numba import njit, prange
from scipy import sparse

from salted import get_averages
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx,
    load_overlap_packed,
    read_system,
)


@njit(parallel=True, cache=True)
def _csr_matvec(indptr, indices, data, x, bounds):
    y = np.zeros(indptr.shape[0] - 1)
    for b in prange(bounds.shape[0] - 1): # blocks of rows, one per task
        for i in range(bounds[b], bounds[b + 1]): # rows of the matrix
            acc = 0.0
            for jj in range(indptr[i], indptr[i + 1]): # non-zero entries of the row
                acc += data[jj] * x[indices[jj]] # accumulate the dot product of the row with x
            y[i] = acc
    return y


def csr_matvec(m, x):
    # y = m @ x for a scipy CSR matrix m, on all numba threads
    # The rows are split into blocks of about equal numbers of non-zeros, one block per task.
    if x.shape[0] != m.shape[1]:
        raise ValueError(f"vector of length {x.shape[0]} for a matrix with {m.shape[1]} columns")
    targets = np.linspace(0, m.indptr[-1], 8 * numba.get_num_threads() + 1)
    bounds = np.unique(np.searchsorted(m.indptr, targets))
    bounds[0], bounds[-1] = 0, m.shape[0]
    return _csr_matvec(m.indptr, m.indices, m.data, x, bounds)


@njit(parallel=True, cache=True)
def _packed_symv(ap, x, bounds):
    # y = S @ x for a symmetric S stored as its lower triangle row by row (row i: S[i, 0..i]).
    # Each task handles one block of rows.
    n = x.shape[0]
    nb = bounds.shape[0] - 1
    ybuf = np.zeros((nb, n))
    y = np.zeros(n)
    for b in prange(nb): # blocks of rows, one per task
        for i in range(bounds[b], bounds[b + 1]): # rows of the matrix
            off = i * (i + 1) // 2
            xi = x[i]
            acc = 0.0
            for j in range(i):
                a = ap[off + j]
                acc += a * x[j]
                ybuf[b, j] += a * xi
            y[i] = acc + ap[off + i] * xi
    for j in prange(n): # add to y[j] the column contributions collected by all tasks
        s = 0.0
        for b in range(nb):
            s += ybuf[b, j]
        y[j] += s
    return y


def packed_symv(ap, x):
    # y = S @ x for a symmetric S stored as its lower triangle (ap, from load_overlap_packed), on all
    # numba threads. The rows are split into blocks of about equal numbers of entries, one per task.
    n = x.shape[0]
    if ap.shape[0] != n * (n + 1) // 2:
        raise ValueError(f"vector of length {n} for a packed matrix of {ap.shape[0]} entries")
    starts = np.arange(n + 1, dtype=np.int64)
    starts = starts * (starts + 1) // 2  # position in ap of the first entry of each row
    bounds = np.unique(np.searchsorted(starts, np.linspace(0, starts[-1], 2 * numba.get_num_threads() + 1)))
    bounds[0], bounds[-1] = 0, n
    return _packed_symv(ap, x, bounds)


def _vdot(a, b):
    # dot product without BLAS (since it slows down the numba kernels)
    return np.einsum("i,i->", a, b)


def build():

    inp = ParseConfig().parse_input()
    # frequently used parameters
    saltedname = inp.salted.saltedname
    saltedpath = inp.salted.saltedpath
    saltedtype = inp.salted.saltedtype
    average = inp.system.average
    zeta = inp.gpr.z
    Menv = inp.gpr.Menv
    Ntrain = inp.gpr.Ntrain
    regul = inp.gpr.regul
    gradtol = inp.gpr.gradtol

    comm, size, rank, parallel = detect_mpi()

    # testing aid: SALTED_MINLOSS_TIMERS=1 prints, per rank at the end, where the time of the
    # CG steps goes (seconds summed over the steps, reported in ms per step)
    timing = os.environ.get("SALTED_MINLOSS_TIMERS", "0") == "1"
    t_build = time.perf_counter()
    timers = dict.fromkeys(["psi", "S", "psiT", "wait", "allreduce", "loss", "save"], 0.0)
    ncalls = {"loss": 0, "grad": 0}

    fdir = f"rkhs-vectors_{saltedname}"
    rdir = f"regrdir_{saltedname}"

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = (
        read_system()
    )

    atom_per_spe, natoms_per_spe = get_atom_idx(ndata, natoms, species, atomic_symbols)

    # load average density coefficients if needed
    if average:
        # compute average density coefficients
        if rank == 0:
            get_averages.build()
        if parallel:
            comm.Barrier()
        # load average density coefficients
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(
                os.path.join(
                    saltedpath, "coefficients", "averages", f"averages_{spe}.npy"
                )
            )

    dirpath = os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}")
    if rank == 0 and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    if parallel:
        comm.Barrier()

    # define training set at random
    if Ntrain > ndata:
        if rank == 0:
            raise ValueError(
                f"More training structures {Ntrain=} have been requested "
                f"than are present in the input data {ndata=}."
            )
        else:
            sys.exit()
    dataset = list(range(ndata))
    if inp.gpr.trainsel == "sequential":
        trainrangetot = dataset[:Ntrain]
    elif inp.gpr.trainsel == "random":
        random.Random(3).shuffle(dataset)
        trainrangetot = dataset[:Ntrain]
    else:
        raise ValueError(f"training set selection {inp.gpr.trainsel} not available!")
    if rank == 0:
        np.savetxt(
            osp.join(saltedpath, rdir, f"training_set_N{Ntrain}.txt"),
            trainrangetot,
            fmt="%i",
        )
    # trainrangetot = np.loadtxt("training_set.txt",int)

    # Distribute structures to tasks
    ntraintot = round(inp.gpr.trainfrac * Ntrain)

    if parallel:
        check_MPI_tasks_count(comm, ntraintot, "training structures")
        trainrange = distribute_jobs(comm, trainrangetot[:ntraintot])
        if inp.salted.verbose:
            print(f"Task {rank} handles the following structures: {format_index_ranges(trainrange,True)}", flush=True)
    else:
        trainrange = trainrangetot[:ntraintot]
    ntrain = len(trainrange)

    def precond_func(ovlp_list, psi_list):
        """Compute preconditioning."""
        # never called, written for full overlap matrices, not for the packed ones (load_overlap_packed)

        #        global totsize
        totsize = psi_list[0].shape[1]
        diag_hessian = np.zeros(totsize)

        for iconf in range(ntrain):

            # psi_vector = psi_list[iconf].toarray()
            # ovlp_times_psi = np.dot(ovlp_list[iconf],psi_vector)
            # diag_hessian += 2.0*np.sum(np.multiply(ovlp_times_psi,psi_vector),axis=0)

            ovlp_times_psi = sparse.csc_matrix.dot(psi_list[iconf].T, ovlp_list[iconf])
            temp = np.sum(
                sparse.csc_matrix.multiply(psi_list[iconf].T, ovlp_times_psi), axis=1
            )
            diag_hessian += 2.0 * np.squeeze(np.asarray(temp))

        # del psi_vector

        return diag_hessian

    def curv_func(cg_dire):
        """Compute curvature on the given CG-direction: 2 (B + lambda I) d, summed over all ranks."""

        totsize = psi_par[0].shape[1]

        Ad = np.zeros(totsize)

        if saltedtype=="density":

            for iconf in range(ntrain):
                t0 = time.perf_counter()
                psi_x_dire = csr_matvec(psi_par[iconf], cg_dire) # Psi @ d
                t1 = time.perf_counter()
                ovlp_x_psi = packed_symv(ovlp_list[iconf], psi_x_dire) # S @ Psi @ d
                t2 = time.perf_counter()
                Ad += 2.0 * csr_matvec(psiT_par[iconf], ovlp_x_psi) # Psi^T @ S @ Psi @ d
                t3 = time.perf_counter()
                timers["psi"] += t1 - t0
                timers["S"] += t2 - t1
                timers["psiT"] += t3 - t2

        elif saltedtype=="density-response":

            itot = 0
            for iconf in range(ntrain):
                for icart in ["x","y","z"]:
                    psi_x_dire = csr_matvec(psi_par[itot], cg_dire)
                    Ad += 2.0 * csr_matvec(psiT_par[itot], packed_symv(ovlp_list[iconf], psi_x_dire))
                    itot += 1
        
        if parallel:
            t0 = time.perf_counter()
            if timing:
                comm.Barrier()  # separates waiting for the slowest rank from the sum itself
            t1 = time.perf_counter()
            Ad = comm.allreduce(Ad) * norm + 2.0 * regul * cg_dire
            timers["wait"] += t1 - t0
            timers["allreduce"] += time.perf_counter() - t1
        else:
            Ad *= norm
            Ad += 2.0 * regul * cg_dire

        return Ad

    def grad_func(weights):
        """Gradient of the loss: 2(B + lambda I) w - 2A"""
        ncalls["grad"] += 1 # count the number of calls to the gradient function (for timing)
        return curv_func(weights) - 2.0 * Avec # 2(B + lambda I) w - 2A

    def loss_func(weights, resid):
        """Loss: w^T @ (B + lambda I) @ w - 2 A @ w + C = C - w @ (A + resid/2)
        where resid is minus the gradient at w, as kept up to date by CG"""
        t0 = time.perf_counter()
        loss = Cconst - _vdot(weights, Avec + 0.5 * resid) # C - w @ (A + resid/2)
        timers["loss"] += time.perf_counter() - t0
        ncalls["loss"] += 1
        return loss

    if rank == 0:
        print("loading matrices...")
    ovlp_list = []
    psi_par = []   # Psi and Psi^T in CSR format, for the multithreaded products
    psiT_par = []
    t_setup = time.perf_counter() - t_build
    t_load_S = t_load_psi = 0.0
    for iconf in trainrange:
        t0 = time.perf_counter()
        ovlp_list.append(
            load_overlap_packed(osp.join(saltedpath, "overlaps", f"overlap_conf{iconf}.npy"))
        )
        t1 = time.perf_counter()
        t_load_S += t1 - t0
        # load feature vector as a scipy sparse object
        if saltedtype=="density":
            psi = sparse.load_npz(osp.join(
              saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
            ))
            psi_par.append(psi.tocsr())
            psiT_par.append(psi.T.tocsr())
        elif saltedtype=="density-response":
            for icart in ["x","y","z"]:
                psi = sparse.load_npz(osp.join(
                  saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                ))
                psi_par.append(psi.tocsr())
                psiT_par.append(psi.T.tocsr())
        t_load_psi += time.perf_counter() - t1

    totsize = psi_par[0].shape[1]
    norm = 1.0 / float(ntraintot)

    # Compute the parts of the loss and gradient that do not depend on the weights
    # A = (1/N) sum Psi^T @ S @ c (Avec of hessian_matrix)
    # C = (1/N) sum c^T @ S @ c
    t0 = time.perf_counter()
    Avec = np.zeros(totsize)
    Cconst = 0.0
    itot = 0
    for iconf in range(ntrain):
        if saltedtype=="density":
            ref_coefs = np.load(osp.join(
                saltedpath, "coefficients", f"coefficients_conf{trainrange[iconf]}.npy"
            ))
            if average:
                i = 0
                for iat in range(natoms[trainrange[iconf]]):
                    spe = atomic_symbols[trainrange[iconf]][iat]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                                ref_coefs[i] -= av_coefs[spe][n] # subtract the average density coefficients for the species
                            i += 2*l+1
            targets = [ref_coefs]
        elif saltedtype=="density-response":
            targets = [np.load(osp.join(
                saltedpath, "coefficients", f"{icart}/coefficients_conf{trainrange[iconf]}.npy"
            )) for icart in ["x","y","z"]]
        for ref_coefs in targets:
            ref_projs = packed_symv(ovlp_list[iconf], ref_coefs) # S @ c
            Avec += csr_matvec(psiT_par[itot], ref_projs) # Psi^T @ S @ c
            Cconst += _vdot(ref_coefs, ref_projs) # c^T @ S @ c
            itot += 1
    if parallel:
        Avec = comm.allreduce(Avec)
        Cconst = comm.allreduce(Cconst)
    Avec *= norm
    Cconst *= norm
    t_const = time.perf_counter() - t0

    if rank == 0:
        print(f"problem dimensionality: {totsize}")

    start = time.time()

    # preconditioner
    P = np.ones(totsize) # no preconditioning

    reg_log10_intstr = str(int(np.log10(regul)))  # for consistency

    # testing aids: SALTED_MINLOSS_SAVE_EVERY=<n> keeps a copy of the weights every n CG steps;
    # SALTED_MINLOSS_MAX_STEPS=<n> changes the maximum number of CG steps (default 100000)
    save_every = int(os.environ.get("SALTED_MINLOSS_SAVE_EVERY", "0"))
    max_steps = int(os.environ.get("SALTED_MINLOSS_MAX_STEPS", "100000"))
    stepdir = osp.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}", "minloss_steps")
    if save_every > 0 and rank == 0:
        os.makedirs(stepdir, exist_ok=True)

    def save_step(step, weights):
        np.save(
            osp.join(stepdir, f"weights_N{ntraintot}_reg{reg_log10_intstr}_step{step}.npy"),
            weights,
        )

    init = True
    if inp.gpr.restart:
        wpath = osp.join(
            saltedpath,
            rdir,
            f"M{Menv}_zeta{zeta}",
            f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        dpath = osp.join(
            saltedpath,
            rdir,
            f"M{Menv}_zeta{zeta}",
            f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        rpath = osp.join(
            saltedpath,
            rdir,
            f"M{Menv}_zeta{zeta}",
            f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        if osp.exists(wpath) and osp.exists(dpath) and osp.exists(rpath):
            init = False
            w = np.load(wpath)
            d = np.load(dpath)
            r = np.load(rpath)
            s = np.multiply(P, r)
            delnew = _vdot(r, s)
            loss = loss_func(w, -grad_func(w))  # the saved r is from the step before w
        else:
            # Print a warning and revert to the else behavior
            print(
                "Warning: One or more required files to restart do not exist. Reverting to default initialization."
            )

    if init:
        w = np.ones(totsize) * 1e-04
        r = -grad_func(w)
        loss = loss_func(w, r)
        d = np.multiply(P, r)
        delnew = _vdot(r, d)

    if rank == 0:
        print("minimizing...")
    t_init = time.time() - start  # initial loss and gradient (or restart)
    timers.update(dict.fromkeys(timers, 0.0))  # from here on, only the CG steps
    ncalls.update(dict.fromkeys(ncalls, 0))
    t_loop = time.perf_counter()
    for i in range(max_steps):
        Ad = curv_func(d) # Curvature 2 (B + lambda I) @ d, summed over all ranks
        curv = _vdot(d, Ad)
        alpha = delnew / curv
        w = w + alpha * d
        t0 = time.perf_counter()
        if save_every > 0 and (i + 1) % save_every == 0 and rank == 0:
            save_step(i + 1, w)
        if (i + 1) % 50 == 0 and rank == 0:
            np.save(
                osp.join(
                    saltedpath,
                    rdir,
                    f"M{Menv}_zeta{zeta}",
                    f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                w,
            )
            np.save(
                osp.join(
                    saltedpath,
                    rdir,
                    f"M{Menv}_zeta{zeta}",
                    f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                d,
            )
            np.save(
                osp.join(
                    saltedpath,
                    rdir,
                    f"M{Menv}_zeta{zeta}",
                    f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                r,
            )
        timers["save"] += time.perf_counter() - t0
        if (i+1)%50==0:
            loss_old = loss
            r -= alpha * Ad
            loss = loss_func(w, r)
            if loss > loss_old:
                if rank == 0:
                    print("WARNING: loss function increased, search direction reset as the steepest descent.")
                r = -grad_func(w)  # exact gradient, free of the rounding accumulated in r
                loss = loss_func(w, r)
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.sqrt(_vdot(r, r)):.3e}, loss: {loss:.3e}, time: {time.time()-start:.1f} s", flush=True)
                if np.sqrt(_vdot(r, r)) < gradtol:
                    break
                d = np.multiply(P, r)
                delnew = _vdot(r, d)
            else:
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.sqrt(_vdot(r, r)):.3e}, loss: {loss:.3e}, time: {time.time()-start:.1f} s", flush=True)
                if np.sqrt(_vdot(r, r)) < gradtol:
                    break
                else:
                    s = np.multiply(P, r)
                    delold = delnew.copy()
                    delnew = _vdot(r, s)
                    beta = delnew / delold
                    d = s + beta * d
        else:
            r -= alpha * Ad
            if np.sqrt(_vdot(r, r)) < gradtol:
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.sqrt(_vdot(r, r)):.3e}, time: {time.time()-start:.1f} s", flush=True)
                break
            else:
                s = np.multiply(P, r)
                delold = delnew.copy()
                delnew = _vdot(r, s)
                beta = delnew / delold
                d = s + beta * d
    t_loop = time.perf_counter() - t_loop

    if timing:
        nsteps = i + 1
        ms = {k: 1e3 * v / nsteps for k, v in timers.items()}
        other = t_loop - sum(timers.values())  # CG vector updates, dot products, gradient norms
        gb = sum(o.nbytes for o in ovlp_list) / 1e9
        rate = f"{gb * nsteps / timers['S']:.1f} GB/s" if timers["S"] > 0 else "-"
        print(
            f"timers rank {rank}: setup {t_setup:.1f} s, load S {t_load_S:.1f} s ({gb:.2f} GB), "
            f"load psi {t_load_psi:.1f} s, A and C (with numba compilation) {t_const:.1f} s, "
            f"initial loss and gradient {t_init:.1f} s, "
            f"{nsteps} steps in {t_loop:.1f} s = {1e3 * t_loop / nsteps:.2f} ms/step: "
            f"psi {ms['psi']:.2f}, S {ms['S']:.2f}, psiT {ms['psiT']:.2f}, wait {ms['wait']:.2f}, "
            f"allreduce {ms['allreduce']:.2f}, loss {ms['loss']:.2f} ({ncalls['loss']} calls), "
            f"grad {ncalls['grad']} calls (time in psi, S, psiT), save {ms['save']:.2f}, "
            f"other {1e3 * other / nsteps:.2f}; S read at {rate}",
            flush=True,
        )

    if rank == 0:
        if save_every > 0:
            save_step(i + 1, w)  # the last step, where the loop stopped
        np.save(
            osp.join(
                saltedpath,
                rdir,
                f"M{Menv}_zeta{zeta}",
                f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            w,
        )
        np.save(
            osp.join(
                saltedpath,
                rdir,
                f"M{Menv}_zeta{zeta}",
                f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            d,
        )
        np.save(
            osp.join(
                saltedpath,
                rdir,
                f"M{Menv}_zeta{zeta}",
                f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            r,
        )
        print("minimization completed succesfully!")
        print(f"minimization time: {((time.time()-start)/60):.2f} minutes")


if __name__ == "__main__":
    build()
