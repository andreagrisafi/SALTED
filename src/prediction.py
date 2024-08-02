import os
import numpy as np
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system,get_conf_range

if inp.parallel:
    from mpi4py import MPI

    # MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size)
else:
    rank = 0

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(inp.predict_filename)

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

kdir = inp.predict_kerndir
rdir = inp.regrdir
pdir = inp.predict_coefdir

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

if rank == 0:
    dirpath = os.path.join(inp.path2qm, pdir)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True) 
    dirpath = os.path.join(inp.path2qm+pdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

ntrain = int(inp.Ntrain*inp.trainfrac)

dirpath = os.path.join(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N_"+str(ntrain))
if rank == 0:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True) 

# load regression weights
weights = np.load(inp.path2ml+rdir+"weights_N"+str(ntrain)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy")

ntest = ndata
testrangetot = np.arange(ndata)

if inp.parallel:
    testrange = get_conf_range(rank,size,ntest,testrangetot)
#    if rank == 0:
#        testrange = [[] for _ in range(size)]
#        blocksize = int(ntest/float(size))
#        for i in range(size):
#            if i == (size-1):
#                testrange[i] = testrangetot[i*blocksize:ntest]
#            else:
#                testrange[i] = testrangetot[i*blocksize:(i+1)*blocksize]
#    else:
#        testrange = None

    testrange = comm.scatter(testrange,root=0)
    print('Task',rank+1,'handles the following structures:',testrange,flush=True)
else:
    testrange = testrangetot[:ntest]

# compute error over test set
error_density = 0
variance = 0
for itest in testrange:

    Tsize = 0
    for iat in range(natoms[itest]):
        spe = atomic_symbols[itest][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Tsize += 2*l+1

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    #print(weights.shape)
    for spe in spelist:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                #rc = 6.0#orcuts[iii]
                #psi_nm = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(itest)+".npy") 
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(itest)+".npy") 
                Mcut = psi_nm.shape[1]
                #print(spe,l,n,Mcut,isize)
                C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                isize += Mcut
                iii += 1
    
        
    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in range(natoms[itest]):
        spe = atomic_symbols[itest][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                if l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1

    # add the average spherical coefficients to the predictions 
    pred_coefs += Av_coeffs

    # save predicted coefficients
    np.save(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(itest)+".npy",pred_coefs)

if (rank == 0): print('Prediction complete')
