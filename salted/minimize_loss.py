import os
import os.path as osp
import random
import sys
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
    timers = dict.fromkeys(["psi", "S", "psiT", "wait", "allreduce", "loss", "grad", "save"], 0.0)
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

    def loss_func(weights, ovlp_list, psi_list):
        """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""

        t0 = time.perf_counter()
        #        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        if saltedtype=="density":

            loss = 0.0
            # loop over training structures
            for iconf in range(ntrain):

                ref_coefs = np.load(
                    osp.join(
                        saltedpath,
                        "coefficients",
                        f"coefficients_conf{trainrange[iconf]}.npy",
                    )
                )

                if average:
                    Av_coeffs = np.zeros(ref_coefs.shape[0])
                i = 0
                for iat in range(natoms[trainrange[iconf]]):
                    spe = atomic_symbols[trainrange[iconf]][iat]
                    for l in range(lmax[spe] + 1):
                        for n in range(nmax[(spe, l)]):
                            if average and l == 0:
                                Av_coeffs[i] = av_coefs[spe][n]
                            i += 2 * l + 1

                # rebuild predicted coefficients
                pred_coefs = sparse.csr_matrix.dot(psi_list[iconf], weights)
                if average:
                    pred_coefs += Av_coeffs

                # compute predicted density projections
                ovlp = ovlp_list[iconf]
                ref_projs = np.dot(ovlp, ref_coefs)
                pred_projs = np.dot(ovlp, pred_coefs)

                # collect gradient contributions
                loss += sparse.csc_matrix.dot(
                    pred_coefs - ref_coefs, pred_projs - ref_projs
                )

        elif saltedtype=="density-response":

            loss = 0.0
            # loop over training structures
            itot = 0
            for iconf in range(ntrain):

                ovlp = ovlp_list[iconf]

                for icart in ["x","y","z"]:

                    ref_coefs = np.load(
                        osp.join(
                            saltedpath,
                            f"coefficients/{icart}/",
                            f"coefficients_conf{trainrange[iconf]}.npy",
                        )
                    )

                    # rebuild predicted coefficients
                    pred_coefs = sparse.csr_matrix.dot(psi_list[itot], weights)

                    # compute predicted density projections
                    ref_projs = np.dot(ovlp, ref_coefs)
                    pred_projs = np.dot(ovlp, pred_coefs)

                    # collect gradient contributions
                    loss += sparse.csc_matrix.dot(
                        pred_coefs - ref_coefs, pred_projs - ref_projs
                    )
                    itot += 1

        loss *= norm
        if parallel:
            loss = comm.allreduce(loss)

        # add regularization term
        loss += regul * np.dot(weights, weights)

        timers["loss"] += time.perf_counter() - t0
        ncalls["loss"] += 1
        return loss

    def grad_func(weights, ovlp_list, psi_list):
        """
        Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function.
        """

        t0 = time.perf_counter()
        #        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        if saltedtype=="density":

            # loop over training structures
            for iconf in range(ntrain):

                # load reference QM data
                ref_coefs = np.load(osp.join(
                    saltedpath, "coefficients", f"coefficients_conf{trainrange[iconf]}.npy"
                ))

                if average:
                    Av_coeffs = np.zeros(ref_coefs.shape[0])
                i = 0
                for iat in range(natoms[trainrange[iconf]]):
                    spe = atomic_symbols[trainrange[iconf]][iat]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if average and l==0:
                                Av_coeffs[i] = av_coefs[spe][n]
                            i += 2*l+1

                # rebuild predicted coefficients
                pred_coefs = sparse.csr_matrix.dot(psi_list[iconf],weights)
                if average:
                    pred_coefs += Av_coeffs

                # compute predicted density projections
                ovlp = ovlp_list[iconf]
                ref_projs = np.dot(ovlp,ref_coefs)
                pred_projs = np.dot(ovlp,pred_coefs)

                # collect gradient contributions
                gradient += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,pred_projs-ref_projs)
        
        elif saltedtype=="density-response":

            # loop over training structures
            itot = 0
            for iconf in range(ntrain):

                ovlp = ovlp_list[iconf]
 
                for icart in ["x","y","z"]:

                    # load reference QM data
                    ref_coefs = np.load(osp.join(
                        saltedpath, "coefficients", f"{icart}/coefficients_conf{trainrange[iconf]}.npy"
                    ))

                    # rebuild predicted coefficients
                    pred_coefs = sparse.csr_matrix.dot(psi_list[itot],weights)

                    # compute predicted density projections
                    ref_projs = np.dot(ovlp,ref_coefs)
                    pred_projs = np.dot(ovlp,pred_coefs)

                    # collect gradient contributions
                    gradient += 2.0 * sparse.csc_matrix.dot(psi_list[itot].T,pred_projs-ref_projs)
                    
                    itot += 1

        if parallel:
            gradient = comm.allreduce(gradient) * norm + 2.0 * regul * weights
        else:
            gradient *= norm
            gradient += 2.0 * regul * weights
        timers["grad"] += time.perf_counter() - t0
        ncalls["grad"] += 1
        return gradient

    def precond_func(ovlp_list, psi_list):
        """Compute preconditioning."""

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

    def curv_func(cg_dire, ovlp_list, psi_list):
        """Compute curvature on the given CG-direction."""

        totsize = psi_list[0].shape[1]

        Ad = np.zeros(totsize)

        if saltedtype=="density":

            for iconf in range(ntrain):
                t0 = time.perf_counter()
                psi_x_dire = sparse.csr_matrix.dot(psi_list[iconf],cg_dire)
                t1 = time.perf_counter()
                ovlp_x_psi = np.dot(ovlp_list[iconf],psi_x_dire)
                t2 = time.perf_counter()
                Ad += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,ovlp_x_psi)
                t3 = time.perf_counter()
                timers["psi"] += t1 - t0
                timers["S"] += t2 - t1
                timers["psiT"] += t3 - t2

        elif saltedtype=="density-response":

            itot = 0
            for iconf in range(ntrain):
                for icart in ["x","y","z"]:
                    psi_x_dire = sparse.csr_matrix.dot(psi_list[itot],cg_dire)
                    Ad += 2.0 * sparse.csc_matrix.dot(psi_list[itot].T,np.dot(ovlp_list[iconf],psi_x_dire))
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

    if rank == 0:
        print("loading matrices...")
    ovlp_list = []
    psi_list = []
    t_setup = time.perf_counter() - t_build
    t_load_S = t_load_psi = 0.0
    for iconf in trainrange:
        t0 = time.perf_counter()
        ovlp_list.append(
            load_overlap(osp.join(saltedpath, "overlaps", f"overlap_conf{iconf}.npy"))
        )
        t1 = time.perf_counter()
        t_load_S += t1 - t0
        # load feature vector as a scipy sparse object
        if saltedtype=="density":
            psi_list.append(sparse.load_npz(osp.join(
              saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
            )))
        elif saltedtype=="density-response":
            for icart in ["x","y","z"]:
                psi_list.append(sparse.load_npz(osp.join(
                  saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                )))
        t_load_psi += time.perf_counter() - t1

    totsize = psi_list[0].shape[1]
    norm = 1.0 / float(ntraintot)

    if rank == 0:
        print(f"problem dimensionality: {totsize}")

    start = time.time()

    # preconditioner
    P = np.ones(totsize)

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
            delnew = np.dot(r, s)
            loss = loss_func(w, ovlp_list, psi_list)
        else:
            # Print a warning and revert to the else behavior
            print(
                "Warning: One or more required files to restart do not exist. Reverting to default initialization."
            )

    if init:
        w = np.ones(totsize) * 1e-04
        loss = loss_func(w, ovlp_list, psi_list)
        r = -grad_func(w, ovlp_list, psi_list)
        d = np.multiply(P, r)
        delnew = np.dot(r, d)

    if rank == 0:
        print("minimizing...")
    t_init = time.time() - start  # initial loss and gradient (or restart)
    timers.update(dict.fromkeys(timers, 0.0))  # from here on, only the CG steps
    ncalls.update(dict.fromkeys(ncalls, 0))
    t_loop = time.perf_counter()
    for i in range(max_steps):
        #loss = loss_func(w, ovlp_list, psi_list)
        Ad = curv_func(d, ovlp_list, psi_list)
        curv = np.dot(d, Ad)
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
            loss_old = loss.copy()
            loss = loss_func(w, ovlp_list, psi_list)
            if loss>loss_old:
                if rank == 0:
                    print("WARNING: loss function increased, search direction reset as the steepest descent.")
                r = -grad_func(w, ovlp_list, psi_list)
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}, loss: {loss:.3e}, time: {time.time()-start:.1f} s", flush=True)
                if np.linalg.norm(r) < gradtol:
                    break
                d = np.multiply(P, r)
                delnew = np.dot(r, d)
            else:
                r -= alpha * Ad
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}, loss: {loss:.3e}, time: {time.time()-start:.1f} s", flush=True)
                if np.linalg.norm(r) < gradtol:
                    break
                else:
                    s = np.multiply(P, r)
                    delold = delnew.copy()
                    delnew = np.dot(r, s)
                    beta = delnew / delold
                    d = s + beta * d
        else:
            r -= alpha * Ad
            if np.linalg.norm(r) < gradtol:
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}, time: {time.time()-start:.1f} s", flush=True)
                break
            else:
                s = np.multiply(P, r)
                delold = delnew.copy()
                delnew = np.dot(r, s)
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
            f"load psi {t_load_psi:.1f} s, initial loss and gradient {t_init:.1f} s; "
            f"{nsteps} steps in {t_loop:.1f} s = {1e3 * t_loop / nsteps:.2f} ms/step: "
            f"psi {ms['psi']:.2f}, S {ms['S']:.2f}, psiT {ms['psiT']:.2f}, wait {ms['wait']:.2f}, "
            f"allreduce {ms['allreduce']:.2f}, loss {ms['loss']:.2f} ({ncalls['loss']} calls), "
            f"grad {ms['grad']:.2f} ({ncalls['grad']} calls), save {ms['save']:.2f}, "
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
