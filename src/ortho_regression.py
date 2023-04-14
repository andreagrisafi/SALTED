import os
import numpy as np
import random
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system, get_atom_idx

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,spelist,atomic_symbols)

# number of sparse environments
M = inp.Menv
# number of training configurations 
N = inp.Ntrain
# training set fraction
frac = inp.trainfrac
# number of sparse environments
reg = inp.regul
eigcut = inp.eigcut
kdir = inp.kerndir
pdir = inp.valcdir

dirpath = os.path.join(inp.path2qm, pdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm+pdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# training set selection
dataset = list(range(ndata))
random.Random(3).shuffle(dataset)
trainrangetot = dataset[:N]
np.savetxt("training_set.txt",trainrangetot,fmt='%i')
#trainrangetot = np.loadtxt("training_set2.txt",int)
ntrain = int(frac*len(trainrangetot))
trainrange = trainrangetot[0:ntrain]
natoms_train = natoms[trainrange]
print("Number of training configurations =", ntrain)
testrange = np.setdiff1d(list(range(ndata)),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]

ortho_preds = np.zeros((ntest,natmax,llmax+1,nnmax,2*llmax+1))
for spe in spelist:

    for l in range(lmax[spe]+1):
      
        # get truncated size
        Mcut = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(0)+".npy").shape[1]
        # compute B matrix
        B = np.zeros((Mcut,Mcut))
        for iconf in trainrange:
            psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy")
            B += np.dot(psi_nm.T,psi_nm)
        B /= ntrain
        
        for n in range(nmax[(spe,l)]): 
        
            # compute A vector
            A = np.zeros(Mcut)
            for iconf in trainrange:
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy")
                ortho_projs = np.load(inp.path2qm+inp.projdir+"spe"+str(spe)+"_l"+str(l)+"_n"+str(n)+"/ortho_projections_conf"+str(iconf)+".npy")
                
                A += np.dot(psi_nm.T,ortho_projs)
            A /= ntrain

            print("")
            print("spe:",spe,"L:",l,"n:",n)
            print("------------------------")
            
            x = np.linalg.solve( B + reg*np.eye(Mcut) , A )

            error_total = 0 
            variance = 0
            itest = 0
            for iconf in testrange:

                # predict
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy")
                ortho_projs = np.dot(psi_nm,x)
              
                # reference
                ortho_projs_ref = np.load(inp.path2qm+inp.projdir+"spe"+str(spe)+"_l"+str(l)+"_n"+str(n)+"/ortho_projections_conf"+str(iconf)+".npy")

                # compute error
                delta = ortho_projs-ortho_projs_ref
                error_total += np.dot(delta,delta)
                variance += np.dot(ortho_projs_ref,ortho_projs_ref)
                #print iconf+1, ":", np.sqrt(error/var)*100, "% RMSE"

                i = 0
                for iat in atom_idx[(iconf,spe)]:
                    for im in range(2*l+1):
                        ortho_preds[itest,iat,l,n,im] = ortho_projs.reshape(len(atom_idx[(iconf,spe)]),2*l+1)[i,im]
                    i+=1
                itest += 1

            print("% RMSE =", 100*np.sqrt(error_total/variance))

np.save(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/ortho-predictions_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",ortho_preds)
