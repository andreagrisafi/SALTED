import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import random
from random import shuffle

from scipy.optimize import minimize

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
    for l in xrange(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut

# paths to data
kdir = inp.kerndir
hdir = inp.hessdir
pdir = inp.preddir

# species dependent arrays
atoms_per_spe = {}
natoms_per_spe = {}
for iconf in xrange(ndata):
    for spe in spelist:
        atoms_per_spe[(iconf,spe)] = []
        natoms_per_spe[(iconf,spe)] = 0

atomic_symbols = []
valences = []
natoms = np.zeros(ndata,int)
for iconf in xrange(ndata):
    atomic_symbols.append(xyzfile[iconf].get_chemical_symbols())
    valences.append(xyzfile[iconf].get_atomic_numbers())
    natoms[iconf] = int(len(atomic_symbols[iconf]))
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        atoms_per_spe[(iconf,spe)].append(iat)
        natoms_per_spe[(iconf,spe)] += 1
natmax = max(natoms)

# load average density coefficients
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# define training set at random
dataset = range(ndata)
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
ntrain = int(inp.trainfrac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]


def grad_func(weights,ovlp_list,psi_list):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    global totsize 

#    start = time.time()

    # init gradient
    gradient = np.zeros(totsize)

    # loop over training structures
    for iconf in xrange(ntrain):
   
#        print iconf
        # load reference QM data
        ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(trainrange[iconf])+".npy")
        ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(trainrange[iconf])+".npy")
       
        Av_coeffs = np.zeros(ref_coefs.shape[0])
        i = 0
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in xrange(lmax[spe]+1):
                for n in xrange(nmax[(spe,l)]):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1

        # rebuild predicted coefficients
        pred_coefs = np.dot(psi_list[iconf],weights)
        pred_coefs += Av_coeffs

        # compute predicted density projections
        pred_projs = np.dot(ovlp_list[iconf],pred_coefs)

        # collect gradient contributions
        gradient += 2.0 * np.dot(psi_list[iconf].T,pred_projs-ref_projs)

    gradient /= ntrain

    # add regularization term
    gradient += 2.0 * inp.regul * weights

#    print "time gradient:", time.time()-start

    return gradient

def precond_func(ovlp_list,psi_list):
    """Compute preconditioning."""

    global totsize

    diag_hessian = np.zeros((totsize))
    
    for iconf in xrange(ntrain):
        print iconf
        for m in xrange(totsize):
            psi_vector_m = psi_list[iconf][:,m]
            diag_hessian[m] += 2.0 * np.dot(psi_vector_m,np.dot(ovlp_list[iconf],psi_vector_m))
  
    diag_hessian /= ntrain
    diag_hessian += 2.0 * inp.regul * np.ones(totsize) 

    return 1.0/diag_hessian 

def curv_func(cg_dire,ovlp_list,psi_list):
    """Compute curvature on the given CG-direction."""
  
    global totsize

#    start = time.time()
    Ap = np.zeros((totsize))

    for iconf in xrange(ntrain):
#        print iconf
        psi_x_dire = np.dot(psi_list[iconf],cg_dire)
        Ap += 2.0 * np.dot(psi_list[iconf].T,np.dot(ovlp_list[iconf],psi_x_dire))
   
    Ap /= ntrain
    # add regularization term
    Ap += 2.0 * inp.regul * cg_dire
#    print "time curvature:", time.time()-start

    return Ap

print "loading matrices..."
ovlp_list = [] 
psi_list = [] 
for iconf in trainrange:
    print iconf
    ovlp_list.append(np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy"))
    psi_list.append(np.load(inp.path2ml+"psi-vectors/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy"))

totsize = psi_list[0].shape[1]
print "problem dimensionality:", totsize

start = time.time()
tol = inp.gradtol 
w = np.ones(totsize)*1e-04
r = - grad_func(w,ovlp_list,psi_list)
print "computing preconditioning matrix..."
P = precond_func(ovlp_list,psi_list)
d = np.multiply(P,r)
delnew = np.dot(r,d)

print "minimizing..."
for i in xrange(2000):
    Ad = curv_func(d,ovlp_list,psi_list)
    curv = np.dot(d,Ad)
    alpha = delnew/curv
    w = w + alpha*d
    r -= alpha * Ad 
    print i+1, "gradient norm:", np.sqrt(np.sum((r**2)))
    if np.sqrt(np.sum((r**2))) < tol:
        break
    else:
        s = np.multiply(P,r)
        delold = delnew.copy()
        delnew = np.dot(r,s)
        beta = delnew/delold
        d = s + beta*d

np.save("weights_N"+str(ntrain)+".npy",w)

print "minimization compleated succesfully!"
print "minimization time:", (time.time()-start)/60, "minutes"


