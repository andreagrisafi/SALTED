import os
import numpy as np
import time
import random
from scipy import sparse
from sys_utils import read_system, get_atom_idx,get_conf_range
import sys
sys.path.insert(0, './')
import inp

# MPI information
if inp.parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size,flush=True)
else:
    rank=0

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul
rdir = inp.regrdir
fdir = inp.featdir

projdir = inp.projdir
coefdir = inp.coefdir
ovlpdir = inp.ovlpdir

atom_per_spe, natom_per_spe = get_atom_idx(ndata,natoms,spelist,atomic_symbols)

# load average density coefficients
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# define training set at random
dataset = list(range(ndata))
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
#trainrangetot = np.loadtxt("training_set.txt",int)

# Distribute structures to tasks
ntraintot = int(inp.trainfrac*inp.Ntrain)

if inp.parallel:
    if rank == 0 and ntraintot < size:
        print('You have requested more processes than training structures. Please reduce the number of processes',flush=True)
        comm.Abort()
    trainrange = get_conf_range(rank,size,ntraintot,trainrangetot)
#    if rank == 0:
#        trainrange = [[] for _ in range(size)]
#        blocksize = int(ntraintot/float(size))
#        for i in range(size):
#            if i == (size-1):
#                trainrange[i] = trainrangetot[i*blocksize:ntraintot]
#            else:
#                trainrange[i] = trainrangetot[i*blocksize:(i+1)*blocksize]
#    else:
#        trainrange = None

    trainrange = comm.scatter(trainrange,root=0)
    print('Task',rank+1,'handles the following structures:',trainrange,flush=True)
else:
    trainrange = trainrangetot[:ntraintot]
ntrain = int(len(trainrange))


def loss_func(weights,ovlp_list,psi_list):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    global totsize 

    # init gradient
    gradient = np.zeros(totsize)

    loss = 0.0
    # loop over training structures
    for iconf in range(ntrain):
   
        # load reference QM data
        ref_projs = np.load(inp.path2qm+projdir+"projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.path2qm+coefdir+"coefficients_conf"+str(trainrange[iconf])+".npy")
       
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms[trainrange[iconf]]):
            spe = atomic_symbols[trainrange[iconf]][iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1

        # rebuild predicted coefficients
        pred_coefs = sparse.csr_matrix.dot(psi_list[iconf],weights)
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


def grad_func(weights,ovlp_list,psi_list):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    global totsize 

    # init gradient
    gradient = np.zeros(totsize)

    # loop over training structures
    for iconf in range(ntrain):
   
        # load reference QM data
        ref_projs = np.load(inp.path2qm+projdir+"projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.path2qm+coefdir+"coefficients_conf"+str(trainrange[iconf])+".npy")
       
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms[trainrange[iconf]]):
            spe = atomic_symbols[trainrange[iconf]][iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1

        # rebuild predicted coefficients
        pred_coefs = sparse.csr_matrix.dot(psi_list[iconf],weights)
        pred_coefs += Av_coeffs

        # compute predicted density projections
        ovlp = ovlp_list[iconf]
        pred_projs = np.dot(ovlp,pred_coefs)

        # collect gradient contributions
        gradient += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,pred_projs-ref_projs)

    return gradient

def precond_func(ovlp_list,psi_list):
    """Compute preconditioning."""

    global totsize
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
  
    global totsize

    Ap = np.zeros((totsize))

    for iconf in range(ntrain):
        psi_x_dire = sparse.csr_matrix.dot(psi_list[iconf],cg_dire)
        Ap += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,np.dot(ovlp_list[iconf],psi_x_dire))

    return Ap

if rank == 0: print("loading matrices...")
ovlp_list = [] 
psi_list = [] 
for iconf in trainrange:
    ovlp_list.append(np.load(inp.path2qm+ovlpdir+"overlap_conf"+str(iconf)+".npy"))
    # load feature vector as a scipy sparse object
    psi_list.append(sparse.load_npz(inp.path2ml+fdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz"))

totsize = psi_list[0].shape[1]
norm = 1.0/float(ntraintot)

if rank == 0: 
    print("problem dimensionality:", totsize)
    dirpath = os.path.join(inp.path2ml, rdir)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


start = time.time()

tol = inp.gradtol 

# preconditioner
P = np.ones(totsize)

if inp.restart == True:
    w = np.load(inp.path2ml+rdir+"weights_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy")
    d = np.load(inp.path2ml+rdir+"dvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy")
    r = np.load(inp.path2ml+rdir+"rvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy")
    s = np.multiply(P,r)
    delnew = np.dot(r,s)
else:
    w = np.ones(totsize)*1e-04
    r = - grad_func(w,ovlp_list,psi_list)
    if inp.parallel:
        r = comm.allreduce(r)*norm  + 2.0 * reg * w
    else:
        r *= norm
        r += 2.0*reg*w
    d = np.multiply(P,r)
    delnew = np.dot(r,d)

if rank == 0: print("minimizing...")
for i in range(100000):
    Ad = curv_func(d,ovlp_list,psi_list)
    if inp.parallel:
        Ad = comm.allreduce(Ad)*norm + 2.0 * reg * d 
    else:
        Ad *= norm
        Ad += 2.0*reg*d
    curv = np.dot(d,Ad)
    alpha = delnew/curv
    w = w + alpha*d
    if (i+1)%50==0 and rank==0:
        np.save(inp.path2ml+rdir+"weights_N"+str(ntraintot)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy",w)
        np.save(inp.path2ml+rdir+"dvector_N"+str(ntraintot)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy",d)
        np.save(inp.path2ml+rdir+"rvector_N"+str(ntraintot)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy",r) 
    r -= alpha * Ad 
    if rank == 0: print(i+1, "gradient norm:", np.sqrt(np.sum((r**2))),flush=True)
    if np.sqrt(np.sum((r**2))) < tol:
        break
    else:
        s = np.multiply(P,r)
        delold = delnew.copy()
        delnew = np.dot(r,s)
        beta = delnew/delold
        d = s + beta*d

if rank == 0:
    np.save(inp.path2ml+rdir+"weights_N"+str(ntraintot)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy",w)
    np.save(inp.path2ml+rdir+"dvector_N"+str(ntraintot)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy",d)
    np.save(inp.path2ml+rdir+"rvector_N"+str(ntraintot)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy",r) 
    print("minimization compleated succesfully!")
    print("minimization time:", (time.time()-start)/60, "minutes")
