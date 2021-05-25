import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse
import random

import basis

from lib import matrices 

sys.path.insert(0, './')
import inp


# read species
spelist = inp.species

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

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
# number of sparse environments
M = inp.Menv
# number of training configurations 
N = inp.Ntrain
# training set fraction
frac = inp.trainfrac
# number of sparse environments
reg = inp.regul
# number of sparse environments
jit = inp.jitter

natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols = xyzfile[i].get_chemical_symbols()
    natoms[i] = int(len(atomic_symbols))
natmax = max(natoms)

atom_idx = {}
for iconf in xrange(ndata):
    for spe in spelist:
        atom_idx[(iconf,spe)] = []

# extract species-dependent power spectrum for lambda=0
for iconf in xrange(ndata):
    atomic_symbols = xyzfile[iconf].get_chemical_symbols()
    for iat in xrange(natoms[iconf]):
        spe = atomic_symbols[iat]
        atom_idx[(iconf,spe)].append(iat)

# training set selection
dataset = range(ndata)
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:N]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
print "Number of training configurations =", ntrain
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]

preds = np.zeros((ntest,natmax,llmax+1,nnmax,2*llmax+1))
for spe in spelist:

    for l in xrange(lmax[spe]+1):
      
        # get truncated size
        Mcut = np.load(inp.path2data+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(0)+"_M"+str(M)+".npy").shape[1]

        # compute B matrix
        B = np.zeros((Mcut,Mcut))
        for iconf in trainrange:
            psi_nm = np.load(inp.path2data+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(iconf)+"_M"+str(M)+".npy")
            B += np.dot(psi_nm.T,psi_nm)
        
        for n in xrange(nmax[(spe,l)]): 
        
            # compute A vector
            A = np.zeros(Mcut)
            for iconf in trainrange:
                psi_nm = np.load(inp.path2data+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(iconf)+"_M"+str(M)+".npy")
                ortho_projs = np.load(inp.path2data+"projections/spe"+str(spe)+"_l"+str(l)+"_n"+str(n)+"/ortho_projections_conf"+str(iconf)+".npy")
                A += np.dot(psi_nm.T,ortho_projs)
           
            print ""
            print "spe:",spe,"L:",l,"n:",n
            print "------------------------"
            
            x = np.linalg.solve( B + reg*np.eye(Mcut) , A )

            error_total = 0 
            variance = 0
            itest = 0
            for iconf in testrange:

                # predict
                psi_nm = np.load(inp.path2data+"kernels/spe"+str(spe)+"_l"+str(l)+"/psi-nm_conf"+str(iconf)+"_M"+str(M)+".npy")
                ortho_projs = np.dot(psi_nm,x)
              
                # reference
                ortho_projs_ref = np.load(inp.path2data+"projections/spe"+str(spe)+"_l"+str(l)+"_n"+str(n)+"/ortho_projections_conf"+str(iconf)+".npy")

                # compute error
                delta = ortho_projs-ortho_projs_ref
                error_total += np.dot(delta,delta)
                variance += np.dot(ortho_projs_ref,ortho_projs_ref)
                #print iconf+1, ":", np.sqrt(error/var)*100, "% RMSE"

                i = 0
                for iat in atom_idx[(iconf,spe)]:
                    for im in xrange(2*l+1):
                        preds[itest,iat,l,n,im] = ortho_projs.reshape(len(atom_idx[(iconf,spe)]),2*l+1)[i,im]
                    i+=1
                itest += 1

            print "% RMSE =", 100*np.sqrt(error_total/variance)

np.save("predictions.npy",preds)
