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

# compute the weight-vector size 
Mcut = {}
totsize = 0
for spe in spelist:
    for l in xrange(lmax[spe]+1):
        Mcut[(spe,l)] = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
        for n in xrange(nmax[(spe,l)]):
            totsize += Mcut[(spe,l)]

print "problem dimensionality:", totsize

# define training set at random
dataset = range(ndata)
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
ntrain = int(inp.trainfrac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]

icount = 0

def loss_func(weights):
    """Given the weight-vector of the RKHS, compute the electron-density loss function."""
 
    global icount 
    icount += 1

    start = time.time()

    # init loss function
    loss = 0.0
   
    # loop over training structures 
    for iconf in trainrange:

        print iconf

        # load reference QM data
        ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
        ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
        overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
        Tsize = len(ref_coefs)
    
        # compute predictions per channel
        C = {}
        ispe = {}
        isize = 0
        for spe in spelist:
            ispe[spe] = 0
            for l in xrange(lmax[spe]+1):
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                for n in xrange(nmax[(spe,l)]):
                    C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                    isize += Mcut
            
        # fill in vector of predictions
        pred_coefs = np.zeros(Tsize)
        Av_coeffs = np.zeros(Tsize)
        i = 0
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in xrange(lmax[spe]+1):
                i1 = ispe[spe]*(2*l+1)
                i2 = ispe[spe]*(2*l+1)+2*l+1
                for n in xrange(nmax[(spe,l)]):
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][i1:i2] 
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1
    
        # rebuild predicted coefficients 
        pred_coefs += Av_coeffs

        # compute predicted density projections
        pred_projs = np.dot(overl,pred_coefs)
        
        # collect error contributions 
        loss += np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
    
    loss /= ntrain

    # add regularization term 
    loss += inp.regul * np.dot(weights,weights)
    
    print "time loss:", time.time()-start
    print "loss value =", loss
    print ""

    return loss

def grad_func(weights):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    start = time.time()

    # init gradient
    gradient = np.zeros(totsize)

#    time_loop = 0.0

    # loop over training structures
    for iconf in trainrange:
   
        print iconf
        # load reference QM data
        ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
        ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
        overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
        Tsize = len(ref_coefs)
       
        start1 = time.time()

        # initialize RKHS feature vectors for each channel 
        Psi = {}
        for spe in spelist:
            for l in xrange(lmax[spe]+1):
                lsize = natoms_per_spe[(iconf,spe)]*(2*l+1) 
                for n in xrange(nmax[(spe,l)]):
                    Psi[(spe,l,n)] = np.zeros((lsize,totsize)) 

        # load the RKHS feature vectors and compute predictions for each channel
        C = {}
        ispe = {}
        isize = 0
        for spe in spelist:
            ispe[spe] = 0
            for l in xrange(lmax[spe]+1):
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                for n in xrange(nmax[(spe,l)]):
                    Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
                    C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                    isize += Mcut
   
        # fill in a single array for RKHS feature vector and predictions
        psi_vector = np.zeros((Tsize,totsize))
        Av_coeffs = np.zeros(Tsize)
        pred_coefs = np.zeros(Tsize)
        i = 0
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in xrange(lmax[spe]+1):
                i1 = ispe[spe]*(2*l+1)
                i2 = ispe[spe]*(2*l+1)+2*l+1
                for n in xrange(nmax[(spe,l)]):
                    psi_vector[i:i+2*l+1] = Psi[(spe,l,n)][i1:i2] 
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][i1:i2] 
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1

#        time_loop += time.time()-start1
        
        # rebuild predicted coefficients
        pred_coefs += Av_coeffs

        # compute predicted density projections
        pred_projs = np.dot(overl,pred_coefs)
    
        # collect gradient contributions
        gradient += 2.0 * np.dot(psi_vector.T,pred_projs-ref_projs)
   
    gradient /= ntrain

    # add regularization term
    gradient += 2.0 * inp.regul * weights

#    print "time loop", time_loop
    print "time gradient:", time.time()-start

    return gradient

def hess_func(weights):
    """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""
  
    start = time.time()

    # init gradient
    hessian = np.zeros((totsize,totsize))

#    time_loop = 0.0

    # loop over training structures
    for iconf in trainrange:
   
        print iconf
        # load reference QM data
        overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
        Tsize = len(overl)
       
        start1 = time.time()

        # initialize RKHS feature vectors for each channel 
        Psi = {}
        for spe in spelist:
            for l in xrange(lmax[spe]+1):
                lsize = natoms_per_spe[(iconf,spe)]*(2*l+1) 
                for n in xrange(nmax[(spe,l)]):
                    Psi[(spe,l,n)] = np.zeros((lsize,totsize)) 

        # load the RKHS feature vectors and compute predictions for each channel
        C = {}
        ispe = {}
        isize = 0
        for spe in spelist:
            ispe[spe] = 0
            for l in xrange(lmax[spe]+1):
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                for n in xrange(nmax[(spe,l)]):
                    Psi[(spe,l,n)][:,isize:isize+Mcut] = psi_nm
                    isize += Mcut
   
        # fill in a single array for RKHS feature vector and predictions
        psi_vector = np.zeros((Tsize,totsize))
        i = 0
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in xrange(lmax[spe]+1):
                i1 = ispe[spe]*(2*l+1)
                i2 = ispe[spe]*(2*l+1)+2*l+1
                for n in xrange(nmax[(spe,l)]):
                    psi_vector[i:i+2*l+1] = Psi[(spe,l,n)][i1:i2] 
                    i += 2*l+1
            ispe[spe] += 1

#        time_loop += time.time()-start1
        
        # collect gradient contributions
        hessian += 2.0 * np.dot(psi_vector.T,np.dot(overl,psi_vector))
   
    hessian /= ntrain

    # add regularization term
    hessian += 2.0 * inp.regul * np.eye(totsize,totsize) 

#    print "time loop", time_loop
    print "time gradient:", time.time()-start

    return hessian 

# initialize the weight-vector
w0 = np.ones(totsize)*1e-04

# minimize the loss function with precomputed gradient
#res = minimize(loss_func,w0,method='BFGS',jac=grad_func,options={'gtol': 1e-05})
res = minimize(loss_func,w0,method='Newton-CG',jac=grad_func,hess=hess_func,options={'gtol': 1e-06})

print "number of minimization steps:", icount

# get the optimal weights
wopt = res.x

# save
np.save("weights.npy",wopt)

