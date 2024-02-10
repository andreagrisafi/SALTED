"""
TODO:
- stop minimization with explicit message from rank 0
"""

import os
import random
import sys
import time
import os.path as osp

import numpy as np
from scipy import sparse
from salted.sys_utils import read_system, get_atom_idx, get_conf_range

def build():

    sys.path.insert(0, './')
    import inp
    paral = inp.parallel

    # MPI information
    if paral:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(f"This is task {rank+1} of {size}", flush=True)
    else:
        rank=0
        size=1

    if inp.field:
        fdir = f"rkhs-vectors_{inp.saltedname}_field"
        rdir = f"regrdir_{inp.saltedname}_field"
    else:
        fdir = f"rkhs-vectors_{inp.saltedname}"
        rdir = f"regrdir_{inp.saltedname}"

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

    # sparse-GPR parameters
    M = inp.Menv
    zeta = inp.z
    reg = inp.regul

    atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # load average density coefficients if needed
    if inp.average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(f"averages_{spe}.npy")

    dirpath = os.path.join(inp.saltedpath, rdir, f"M{M}_zeta{zeta}")
    if rank==0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    if size > 1: comm.Barrier()

    # define training set at random
    if (inp.Ntrain > ndata):
        if rank == 0:
            raise ValueError(f"More training structures {inp.Ntrain=} have been requested than are present in the input data {ndata=}.")
        else:
            exit()
    dataset = list(range(ndata))
    random.Random(3).shuffle(dataset)
    trainrangetot = dataset[:inp.Ntrain]
    if rank == 0:
        np.savetxt(osp.join(
            inp.saltedpath, rdir, f"training_set_N{inp.Ntrain}.txt"
        ), trainrangetot, fmt='%i')
    #trainrangetot = np.loadtxt("training_set.txt",int)

    # Distribute structures to tasks
    ntraintot = int(inp.trainfrac*inp.Ntrain)

    if paral:
        if ntraintot < size:
            if rank == 0:
                raise ValueError(f"More processes {size=} have been requested than training structures {ntraintot=}. Please reduce the number of processes.")
            else:
                exit()
        # if rank == 0 and ntraintot < size:
        #     print('You have requested more processes than training structures. Please reduce the number of processes',flush=True)
        #     comm.Abort()
        trainrange = get_conf_range(rank,size,ntraintot,trainrangetot)
        trainrange = comm.scatter(trainrange,root=0)
        print(f"Task {rank+1} handles the following structures: {trainrange}", flush=True)
    else:
        trainrange = trainrangetot[:ntraintot]
    ntrain = int(len(trainrange))


    def loss_func(weights,ovlp_list,psi_list):
        """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""

#        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        loss = 0.0
        # loop over training structures
        for iconf in range(ntrain):

            # load reference QM data
            ref_projs = np.load(osp.join(
                inp.saltedpath, "projections", f"projections_conf{trainrange[iconf]}.npy"
            ))
            ref_coefs = np.load(osp.join(
                inp.saltedpath, "coefficients", f"coefficients_conf{trainrange[iconf]}.npy"
            ))

            if inp.average:
                Av_coeffs = np.zeros(ref_coefs.shape[0])
            i = 0
            for iat in range(natoms[trainrange[iconf]]):
                spe = atomic_symbols[trainrange[iconf]][iat]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if inp.average and l==0:
                            Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1

            # rebuild predicted coefficients
            pred_coefs = sparse.csr_matrix.dot(psi_list[iconf],weights)
            if inp.average:
                pred_coefs += Av_coeffs

            # compute predicted density projections
            ovlp = ovlp_list[iconf]
            pred_projs = np.dot(ovlp,pred_coefs)

            # collect gradient contributions
            loss += sparse.csc_matrix.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)

        loss /= ntrain

        # add regularization term
        loss += reg * np.dot(weights,weights)

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
            ref_projs = np.load(osp.join(
                inp.saltedpath, "projections", f"projections_conf{trainrange[iconf]}.npy"
            ))
            ref_coefs = np.load(osp.join(
                inp.saltedpath, "coefficients", f"coefficients_conf{trainrange[iconf]}.npy"
            ))

            if inp.average:
                Av_coeffs = np.zeros(ref_coefs.shape[0])
            i = 0
            for iat in range(natoms[trainrange[iconf]]):
                spe = atomic_symbols[trainrange[iconf]][iat]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if inp.average and l==0:
                            Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1

            # rebuild predicted coefficients
            pred_coefs = sparse.csr_matrix.dot(psi_list[iconf],weights)
            if inp.average:
                pred_coefs += Av_coeffs

            # compute predicted density projections
            ovlp = ovlp_list[iconf]
            pred_projs = np.dot(ovlp,pred_coefs)

            # collect gradient contributions
            gradient += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,pred_projs-ref_projs)

        return gradient

    def precond_func(ovlp_list,psi_list):
        """Compute preconditioning."""

#        global totsize
        totsize = psi_list[0].shape[1]
        diag_hessian = np.zeros(totsize)

        for iconf in range(ntrain):

            print(iconf+1,flush=True)
            #psi_vector = psi_list[iconf].toarray()
            #ovlp_times_psi = np.dot(ovlp_list[iconf],psi_vector)
            #diag_hessian += 2.0*np.sum(np.multiply(ovlp_times_psi,psi_vector),axis=0)

            ovlp_times_psi = sparse.csc_matrix.dot(psi_list[iconf].T,ovlp_list[iconf])
            temp = np.sum(sparse.csc_matrix.multiply(psi_list[iconf].T,ovlp_times_psi),axis=1)
            diag_hessian += 2.0*np.squeeze(np.asarray(temp))

        #del psi_vector

        return diag_hessian

    def curv_func(cg_dire,ovlp_list,psi_list):
        """Compute curvature on the given CG-direction."""

#        global totsize
        totsize = psi_list[0].shape[1]

        Ap = np.zeros((totsize))

        for iconf in range(ntrain):
            psi_x_dire = sparse.csr_matrix.dot(psi_list[iconf],cg_dire)
            Ap += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,np.dot(ovlp_list[iconf],psi_x_dire))

        return Ap

    if rank == 0:  print("loading matrices...")
    ovlp_list = []
    psi_list = []
    for iconf in trainrange:
        ovlp_list.append(np.load(osp.join(
            inp.saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
        )))
        # load feature vector as a scipy sparse object
        psi_list.append(sparse.load_npz(osp.join(
            inp.saltedpath, fdir, f"M{M}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
        )))

    totsize = psi_list[0].shape[1]
    norm = 1.0 / float(ntraintot)

    if rank == 0:  print(f"problem dimensionality: {totsize}")

    start = time.time()

    tol = inp.gradtol

    # preconditioner
    P = np.ones(totsize)

    reg_log10_intstr = str(int(np.log10(reg)))  # for consistency

    if inp.restart == True:
        w = np.load(osp.join(
            inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy"
        ))
        d = np.load(osp.join(
            inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy"
        ))
        r = np.load(osp.join(
            inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy"
        ))
        s = np.multiply(P,r)
        delnew = np.dot(r,s)
    else:
        w = np.ones(totsize)*1e-04
        r = - grad_func(w,ovlp_list,psi_list)
        if paral:
            r = comm.allreduce(r)*norm  + 2.0 * reg * w
        else:
            r *= norm
            r += 2.0*reg*w
        d = np.multiply(P,r)
        delnew = np.dot(r,d)

    if rank == 0: print("minimizing...")
    for i in range(100000):
        Ad = curv_func(d,ovlp_list,psi_list)
        if paral:
            Ad = comm.allreduce(Ad)*norm + 2.0 * reg * d
        else:
            Ad *= norm
            Ad += 2.0*reg*d
        curv = np.dot(d,Ad)
        alpha = delnew/curv
        w = w + alpha*d
        if (i+1)%50==0 and rank==0:
            np.save(osp.join(
                inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy"
            ), w)
            np.save(osp.join(
                inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy"
            ), d)
            np.save(osp.join(
                inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy"
            ), r)
        r -= alpha * Ad
        if rank == 0:
            # np.sqrt(np.sum((r**2))) == np.linalg.norm(r)
            print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}", flush=True)
        if np.linalg.norm(r) < tol:
            break
        else:
            s = np.multiply(P,r)
            delold = delnew.copy()
            delnew = np.dot(r,s)
            beta = delnew/delold
            d = s + beta*d

    if rank == 0:
        np.save(osp.join(
            inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy"
        ), w)
        np.save(osp.join(
            inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy"
        ), d)
        np.save(osp.join(
            inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy"
        ), r)
        print("minimization compleated succesfully!")
        print(f"minimization time: {((time.time()-start)/60):.2f} minutes")

if __name__ == "__main__":
    build()
