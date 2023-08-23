import os
import numpy as np
import time
import random
from scipy import sparse
from sys_utils import read_system, get_atom_idx, get_conf_range
import sys
sys.path.insert(0, './')
import inp

paral = inp.parallel

# MPI information
if paral:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size,flush=True)
else:
    rank=0

if inp.field:
    fdir = "rkhs-vectors_"+inp.saltedname+"_field"
    rdir = "regrdir_"+inp.saltedname+"_field"
else:
    fdir = "rkhs-vectors_"+inp.saltedname
    rdir = "regrdir_"+inp.saltedname

species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

# sparse-GPR parameters
M = inp.Menv
zeta = inp.z
reg = inp.regul
projdir = inp.projdir
coefdir = inp.coefdir

atom_per_spe, natoms_per_spe = get_atom_idx(ndata,natoms,species,atomic_symbols)

# load average density coefficients if needed
if inp.average:
    av_coefs = {}
    for spe in species:
        av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

if rank==0:
    dirpath = os.path.join(inp.saltedpath, rdir)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    dirpath = os.path.join(inp.saltedpath+rdir+"/", "M"+str(M)+"_zeta"+str(zeta))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

# define training set at random
if (inp.Ntrain > ndata):
    print("More training structures have been requested than are present in the input data.")
    exit()
dataset = list(range(ndata))
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
if rank == 0: np.savetxt(inp.saltedpath+rdir+"/training_set_N"+str(inp.Ntrain)+".txt",trainrangetot,fmt='%i')
#trainrangetot = np.loadtxt("training_set.txt",int)

# Distribute structures to tasks
ntraintot = int(inp.trainfrac*inp.Ntrain)

if paral:
    if rank == 0 and ntraintot < size:
        print('You have requested more processes than training structures. Please reduce the number of processes',flush=True)
        comm.Abort()
    trainrange = get_conf_range(rank,size,ntraintot,trainrangetot)
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
        ref_projs = np.load(inp.saltedpath+projdir+"projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.saltedpath+coefdir+"coefficients_conf"+str(trainrange[iconf])+".npy")
       
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


def grad_func(weights,ovlp_list,psi_list):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    global totsize 

    # init gradient
    gradient = np.zeros(totsize)

    # loop over training structures
    for iconf in range(ntrain):
   
        # load reference QM data
        ref_projs = np.load(inp.saltedpath+projdir+"projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.saltedpath+coefdir+"coefficients_conf"+str(trainrange[iconf])+".npy")
      
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
    ovlp_list.append(np.load(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy"))
    # load feature vector as a scipy sparse object
    psi_list.append(sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npz"))

totsize = psi_list[0].shape[1]
norm = 1.0/float(ntraintot)

if rank == 0: 
    print("problem dimensionality:", totsize)
    dirpath = os.path.join(inp.saltedpath, rdir)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


start = time.time()

tol = inp.gradtol 

# preconditioner
P = np.ones(totsize)

if inp.restart == True:
    w = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy")
    d = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/dvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy")
    r = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/rvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy")
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
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",w)
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/dvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",d)
        np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/rvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",r)
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
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",w)
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/dvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",d)
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/rvector_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",r)
    print("minimization compleated succesfully!")
    print("minimization time:", (time.time()-start)/60, "minutes")
