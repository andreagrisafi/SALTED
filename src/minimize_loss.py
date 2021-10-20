import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle
from scipy import sparse
from scipy.optimize import minimize
from mpi4py import MPI

import basis

sys.path.insert(0, './')
import inp
# MPI information
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('This is task',rank,'of',size)

# system definition
spelist = inp.species
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# basis definition
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

# species dependent arrays
atoms_per_spe = {}
natoms_per_spe = {}
for iconf in range(ndata):
    for spe in spelist:
        atoms_per_spe[(iconf,spe)] = []
        natoms_per_spe[(iconf,spe)] = 0

atomic_symbols = []
valences = []
natoms = np.zeros(ndata,int)
for iconf in range(ndata):
    atomic_symbols.append(xyzfile[iconf].get_chemical_symbols())
    valences.append(xyzfile[iconf].get_atomic_numbers())
    natoms[iconf] = int(len(atomic_symbols[iconf]))
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        atoms_per_spe[(iconf,spe)].append(iat)
        natoms_per_spe[(iconf,spe)] += 1
natmax = max(natoms)

# load average density coefficients
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# define training set at random
#dataset = list(range(ndata))
#random.Random(3).shuffle(dataset)
#trainrangetot = dataset[:inp.Ntrain]
#np.savetxt("training_set.txt",trainrangetot,fmt='%i')
trainrangetot = np.loadtxt("training_set.txt",int)

# Distribute structures to tasks
ntraintot = int(inp.trainfrac*inp.Ntrain)
if rank == 0:
    trainrange = [[] for _ in range(size)]
    blocksize = int(round(ntraintot/np.float(size)))
    for i in range(size):
        if i == (size-1):
            trainrange[i] = trainrangetot[i*blocksize:ntraintot]
        else:
            trainrange[i] = trainrangetot[i*blocksize:(i+1)*blocksize]
else:
    trainrange = None
trainrange = comm.scatter(trainrange,root=0)
ntrain = int(len(trainrange))
print('Task',rank,'handles the following structures:',trainrange)


def loss_func(weights,ovlp_list,psi_list):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    global totsize 

    # init gradient
    gradient = np.zeros(totsize)

    # loop over training structures
    for iconf in range(ntrain):
   
        # load reference QM data
        ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(trainrange[iconf])+".npy")
       
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
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
        ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(trainrange[iconf])+".npy")
       
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
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
    ovlp_list.append(np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy"))
    # load feature vector as a numpy array
#    psi_list.append(np.load(inp.path2ml+psi-vectors/M+str(M)+_eigcut+str(int(np.log10(eigcut)))+/psi-nm_conf+str(iconf)+.npy))
    # load feature vector as a scipy sparse object
    psi_list.append(sparse.load_npz(inp.path2ml+"psi-vectors/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz"))

totsize = psi_list[0].shape[1]
if rank == 0: print("problem dimensionality:", totsize)

start = time.time()

tol = inp.gradtol 

if inp.restart == True:
    w = np.load("weights_N"+str(ntraintot)+".npy")
else:    
    w = np.ones(totsize)*1e-04

r = - grad_func(w,ovlp_list,psi_list)

if rank == 0: print("computing preconditioning matrix...")
#P = precond_func(ovlp_list,psi_list)
#P = comm.allreduce(P)/float(ntraintot) + 2.0 * reg * np.ones(totsize)
#P = 1.0/P
P = np.ones(totsize)

r = comm.allreduce(r)/float(ntraintot) + 2.0 * reg * w

d = np.multiply(P,r)
delnew = np.dot(r,d)

if rank == 0: print("minimizing...")
for i in range(100000):
    Ad = curv_func(d,ovlp_list,psi_list)
    Ad = comm.allreduce(Ad)/float(ntraintot) + 2.0 * reg * d 
    curv = np.dot(d,Ad)
    alpha = delnew/curv
    w = w + alpha*d
    if (i+1)%10==0:
        np.save("weights_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",w) 
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
    np.save("weights_N"+str(ntraintot)+"_reg"+str(int(np.log10(reg)))+".npy",w)
    print("minimization compleated succesfully!")
    print("minimization time:", (time.time()-start)/60, "minutes")
