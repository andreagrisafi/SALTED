"""
TODO:
- stop minimization with explicit message from rank 0
"""

import os
import os.path as osp
import random
import sys
import time

import numpy as np
from scipy import sparse

from salted import get_averages
from salted.sys_utils import ParseConfig, get_atom_idx, get_conf_range, read_system


def build():

    inp = ParseConfig().parse_input()
    (
        saltedname,
        saltedpath,
        saltedtype,
        filename,
        species,
        average,
        field,
        parallel,
        path2qm,
        qmcode,
        qmbasis,
        dfbasis,
        filename_pred,
        predname,
        predict_data,
        rep1,
        rcut1,
        sig1,
        nrad1,
        nang1,
        neighspe1,
        rep2,
        rcut2,
        sig2,
        nrad2,
        nang2,
        neighspe2,
        sparsify,
        nsamples,
        ncut,
        zeta,
        Menv,
        Ntrain,
        trainfrac,
        regul,
        eigcut,
        gradtol,
        restart,
        blocksize,
        trainsel,
        nspe1,
        nspe2,
        HYPER_PARAMETERS_DENSITY,
        HYPER_PARAMETERS_POTENTIAL,
    ) = ParseConfig().get_all_params()

    # MPI information
    if parallel:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(f"This is task {rank+1} of {size}", flush=True)
    else:
        rank = 0
        size = 1

    if field:
        fdir = f"rkhs-vectors_{saltedname}_field"
        rdir = f"regrdir_{saltedname}_field"
    else:
        fdir = f"rkhs-vectors_{saltedname}"
        rdir = f"regrdir_{saltedname}"

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = (
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
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    if size > 1:
        comm.Barrier()

    # define training set at random
    if Ntrain > ndata:
        if rank == 0:
            raise ValueError(
                f"More training structures {Ntrain=} have been requested "
                f"than are present in the input data {ndata=}."
            )
        else:
            exit()
    dataset = list(range(ndata))
    if trainsel == "sequential":
        trainrangetot = dataset[:Ntrain]
    elif trainsel == "random":
        random.Random(3).shuffle(dataset)
        trainrangetot = dataset[:Ntrain]
    else:
        raise ValueError(f"training set selection {trainsel=} not available!")
    if rank == 0:
        np.savetxt(
            osp.join(saltedpath, rdir, f"training_set_N{Ntrain}.txt"),
            trainrangetot,
            fmt="%i",
        )
    # trainrangetot = np.loadtxt("training_set.txt",int)

    # Distribute structures to tasks
    ntraintot = int(trainfrac * Ntrain)

    if parallel:
        if ntraintot < size:
            if rank == 0:
                raise ValueError(
                    f"More processes {size=} have been requested than training structures {ntraintot=}. "
                    f"Please reduce the number of processes."
                )
            else:
                exit()
        # if rank == 0 and ntraintot < size:
        #     print('You have requested more processes than training structures. Please reduce the number of processes',flush=True)
        #     comm.Abort()
        trainrange = get_conf_range(rank, size, ntraintot, trainrangetot)
        trainrange = comm.scatter(trainrange, root=0)
        print(
            f"Task {rank+1} handles the following structures: {trainrange}", flush=True
        )
    else:
        trainrange = trainrangetot[:ntraintot]
    ntrain = int(len(trainrange))

    def loss_func(weights, ovlp_list, psi_list):
        """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""

        #        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        loss = 0.0
        # loop over training structures
        for iconf in range(ntrain):

            # load reference QM data
            ref_projs = np.load(
                osp.join(
                    saltedpath,
                    "projections",
                    f"projections_conf{trainrange[iconf]}.npy",
                )
            )
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
            pred_projs = np.dot(ovlp, pred_coefs)

            # collect gradient contributions
            loss += sparse.csc_matrix.dot(
                pred_coefs - ref_coefs, pred_projs - ref_projs
            )

        loss /= ntrain

        # add regularization term
        loss += regul * np.dot(weights, weights)

        return loss

    def grad_func(weights, ovlp_list, psi_list):
        """
        Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function.
        """

        #        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        # loop over training structures
        for iconf in range(ntrain):

            # load reference QM data
            ref_projs = np.load(
                osp.join(
                    saltedpath,
                    "projections",
                    f"projections_conf{trainrange[iconf]}.npy",
                )
            )
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
            pred_projs = np.dot(ovlp, pred_coefs)

            # collect gradient contributions
            gradient += 2.0 * sparse.csc_matrix.dot(
                psi_list[iconf].T, pred_projs - ref_projs
            )

        return gradient

    def precond_func(ovlp_list, psi_list):
        """Compute preconditioning."""

        #        global totsize
        totsize = psi_list[0].shape[1]
        diag_hessian = np.zeros(totsize)

        for iconf in range(ntrain):

            print(iconf + 1, flush=True)
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

        #        global totsize
        totsize = psi_list[0].shape[1]

        Ap = np.zeros((totsize))

        for iconf in range(ntrain):
            psi_x_dire = sparse.csr_matrix.dot(psi_list[iconf], cg_dire)
            Ap += 2.0 * sparse.csc_matrix.dot(
                psi_list[iconf].T, np.dot(ovlp_list[iconf], psi_x_dire)
            )

        return Ap

    if rank == 0:
        print("loading matrices...")
    ovlp_list = []
    psi_list = []
    for iconf in trainrange:
        ovlp_list.append(
            np.load(osp.join(saltedpath, "overlaps", f"overlap_conf{iconf}.npy"))
        )
        # load feature vector as a scipy sparse object
        psi_list.append(
            sparse.load_npz(
                osp.join(
                    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
                )
            )
        )

    totsize = psi_list[0].shape[1]
    norm = 1.0 / float(ntraintot)

    if rank == 0:
        print(f"problem dimensionality: {totsize}")

    start = time.time()

    # preconditioner
    P = np.ones(totsize)

    reg_log10_intstr = str(int(np.log10(regul)))  # for consistency

    if restart == True:
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
            w = np.load(wpath)
            d = np.load(dpath)
            r = np.load(rpath)
            s = np.multiply(P, r)
            delnew = np.dot(r, s)
        else:
            # Print a warning and revert to the else behavior
            print(
                "Warning: One or more required files to restart do not exist. Reverting to default initialization."
            )
            w = np.ones(totsize) * 1e-04
            r = -grad_func(w, ovlp_list, psi_list)
            if parallel:
                r = comm.allreduce(r) * norm + 2.0 * regul * w
            else:
                r *= norm
                r += 2.0 * regul * w
            d = np.multiply(P, r)
            delnew = np.dot(r, d)
    else:
        w = np.ones(totsize) * 1e-04
        r = -grad_func(w, ovlp_list, psi_list)
        if parallel:
            r = comm.allreduce(r) * norm + 2.0 * regul * w
        else:
            r *= norm
            r += 2.0 * regul * w
        d = np.multiply(P, r)
        delnew = np.dot(r, d)

    if rank == 0:
        print("minimizing...")
    for i in range(100000):
        Ad = curv_func(d, ovlp_list, psi_list)
        if parallel:
            Ad = comm.allreduce(Ad) * norm + 2.0 * regul * d
        else:
            Ad *= norm
            Ad += 2.0 * regul * d
        curv = np.dot(d, Ad)
        alpha = delnew / curv
        w = w + alpha * d
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
        r -= alpha * Ad
        if rank == 0:
            # np.sqrt(np.sum((r**2))) == np.linalg.norm(r)
            print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}", flush=True)
        if np.linalg.norm(r) < gradtol:
            break
        else:
            s = np.multiply(P, r)
            delold = delnew.copy()
            delnew = np.dot(r, s)
            beta = delnew / delold
            d = s + beta * d

    if rank == 0:
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
        print("minimization compleated succesfully!")
        print(f"minimization time: {((time.time()-start)/60):.2f} minutes")


if __name__ == "__main__":
    build()
