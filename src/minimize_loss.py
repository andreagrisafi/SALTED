import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle
from scipy.optimize import minimize
import random

import basis

sys.path.insert(0, './')
import inp

spelist = inp.species
# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in xrange(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
kdir = inp.kerndir
pdir = inp.preddir

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# compute the total size of weight vectors
Mcut = {}
totsize = 0
for spe in spelist:
    for l in xrange(lmax[spe]+1):
        Mcut[(spe,l)] = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
        for n in xrange(nmax[(spe,l)]):
            totsize += Mcut[(spe,l)]

print "problem dimensionality:", totsize

# define training set
dataset = range(ndata)
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:inp.Ntrain]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
ntrain = int(inp.trainfrac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]

def minim_func(weights):
  
    start = time.time()

    #print weights
    loss = 0.0
    
    for iconf in trainrange:

        # load reference
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
            
        # fill vector of predictions
        pred_coefs = np.zeros(Tsize)
        Av_coeffs = np.zeros(Tsize)
        i = 0
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in xrange(lmax[spe]+1):
                for n in xrange(nmax[(spe,l)]):
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1
    
        # rebuild predictions
        pred_coefs += Av_coeffs
        pred_projs = np.dot(overl,pred_coefs)
        
        # collect error
        loss += np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
    
    
    loss += inp.regul * np.dot(weights,weights)
    print "time loss:", time.time()-start
    print loss
    print ""

    return loss

def grad_func(weights):
  
    start = time.time()
    gradient = np.zeros(totsize)

    for iconf in trainrange:
    
        # load reference
        ref_projs = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
        ref_coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
        overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
        Tsize = len(ref_coefs)
        
        Psi = {}
        for spe in spelist:
            for l in xrange(lmax[spe]+1):
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                for n in xrange(nmax[(spe,l)]):
                    Psi[(spe,l,n)] = np.zeros((psi_nm.shape[0],totsize)) 

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
   
        psi_vector = np.zeros((Tsize,totsize))
        Av_coeffs = np.zeros(Tsize)
        pred_coefs = np.zeros(Tsize)
        i = 0
        for iat in xrange(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in xrange(lmax[spe]+1):
                for n in xrange(nmax[(spe,l)]):
                    psi_vector[i:i+2*l+1] = Psi[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1
        
        # rebuild predictions
        pred_coefs += Av_coeffs
        pred_projs = np.dot(overl,pred_coefs)
    
        gradient += 2.0 * np.dot(psi_vector.T,pred_projs-ref_projs)
    
    gradient += 2.0 * inp.regul * weights
    print "time gradient:", time.time()-start

    return gradient

# initialize the weight vector
w0 = np.ones(totsize)*1e-10
res = minimize(minim_func,w0,method='CG',jac=grad_func,options={'gtol': 1e-10})
#res = minimize(minim_func,w0,method='BFGS',jac=grad_func,options={'gtol': 1e-04})
wopt = res.x

np.save("weights.npy",wopt)

