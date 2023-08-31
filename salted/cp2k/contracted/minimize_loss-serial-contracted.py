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

import basis

sys.path.insert(0, './')
import inp

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

pdir = inp.preddir
projdir = inp.projdir
coefdir = inp.coefdir
regrdir = inp.regrdir
featdir = inp.featdir

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

projector = {}
ncut = {}
for spe in spelist:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]

# load average density coefficients
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# define training set at random
dataset = list(range(ndata))
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')

# Distribute structures to tasks
ntrain = int(inp.trainfrac*inp.Ntrain)
trainrange = trainrangetot[:ntrain]

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
                for n in range(ncut[(spe,l)]):
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
                for n in range(ncut[(spe,l)]):
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

def curv_func(cg_dire,ovlp_list,psi_list):
    """Compute curvature on the given CG-direction."""
  
    global totsize

    Ap = np.zeros((totsize))

    for iconf in range(ntrain):
        psi_x_dire = sparse.csr_matrix.dot(psi_list[iconf],cg_dire)
        Ap += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,np.dot(ovlp_list[iconf],psi_x_dire))

    return Ap

print("loading matrices...")
ovlp_list = [] 
psi_list = [] 
for iconf in trainrange:
    ovlp_list.append(np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy"))
    # load feature vector as a numpy array
#    psi_list.append(np.load(inp.path2ml+psi-vectors/M+str(M)+_eigcut+str(int(np.log10(eigcut)))+/psi-nm_conf+str(iconf)+.npy))
    # load feature vector as a scipy sparse object
    psi_list.append(sparse.load_npz(inp.path2ml+featdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npz"))

totsize = psi_list[0].shape[1]
print("problem dimensionality:", totsize,flush=True)

start = time.time()

tol = inp.gradtol 

# save auxiliary projections
dirpath = os.path.join(inp.path2ml, regrdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

P = np.ones(totsize)

if inp.restart == True:
    w = np.load(inp.path2ml+regrdir+"weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
    d = np.load(inp.path2ml+regrdir+"dvector_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
    r = np.load(inp.path2ml+regrdir+"rvector_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
    s = np.multiply(P,r)
    delnew = np.dot(r,s)
else:    
    w = np.ones(totsize)*1e-04
    r = - grad_func(w,ovlp_list,psi_list)/float(ntrain) + 2.0 * reg * w
    d = np.multiply(P,r)
    delnew = np.dot(r,d)

print("minimizing...",flush=True)
for i in range(100000):
    Ad = curv_func(d,ovlp_list,psi_list)/float(ntrain) + 2.0 * reg * d
    curv = np.dot(d,Ad)
    alpha = delnew/curv
    w = w + alpha*d
    if (i+1)%50==0:
        np.save(inp.path2ml+regrdir+"weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",w)
        np.save(inp.path2ml+regrdir+"dvector_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",d)
        np.save(inp.path2ml+regrdir+"rvector_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",r)
    r -= alpha * Ad 
    print(i+1, "gradient norm:", np.sqrt(np.sum((r**2))),flush=True)
    if np.sqrt(np.sum((r**2))) < tol:
        break
    else:
        s = np.multiply(P,r)
        delold = delnew.copy()
        delnew = np.dot(r,s)
        beta = delnew/delold
        d = s + beta*d

np.save(inp.path2ml+regrdir+"weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",w)
np.save(inp.path2ml+regrdir+"dvector_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",d)
np.save(inp.path2ml+regrdir+"rvector_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",r)
print("minimization compleated succesfully!")
print("minimization time:", (time.time()-start)/60, "minutes")
