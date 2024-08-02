import os
import numpy as np
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system,get_conf_range
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r", "--response", action='store_true', help="Specify if validating a field direction other than that used to train the model")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
response = args.response

if inp.parallel:
    from mpi4py import MPI
    # MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('This is task',rank+1,'of',size)
else:
    rank=0

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

projdir = inp.projdir
coefdir = inp.coefdir
ovlpdir = inp.ovlpdir

kdir = inp.kerndir
pdir = inp.valcdir
rdir = inp.regrdir

if response and not os.path.exists("regr_averages_"+str(spelist[0])+".npy"):
    print("The averages used when trining the regression model need to be present, with the prefix 'regr_'") 
if response:
    kdir = inp.predict_kerndir

av_coefs = {}
if response: regr_av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")
    if response: regr_av_coefs[spe] = np.load("regr_averages_"+str(spe)+".npy")

trainrangetot = np.loadtxt("training_set.txt",int)
ntrain = int(inp.trainfrac*len(trainrangetot))
testrangetot = np.setdiff1d(list(range(ndata)),trainrangetot)

if rank == 0:
    dirpath = os.path.join(inp.path2qm, pdir)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    dirpath = os.path.join(inp.path2qm+pdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    dirpath = os.path.join(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N_"+str(ntrain))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

# load regression weights
weights = np.load(inp.path2ml+rdir+"weights_N"+str(ntrain)+"_M"+str(M)+"_reg"+str(int(np.log10(reg)))+".npy")

# compute error over test set
error_density = 0
variance = 0

# Distribute structures to tasks
ntest = len(testrangetot)
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

for iconf in testrange:

    # load reference
    ref_projs = np.load(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy")
    ref_coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    overl = np.load(inp.path2qm+inp.ovlpdir+"overlap_conf"+str(iconf)+".npy")
    Tsize = len(ref_coefs)

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    for spe in spelist:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                #rc = 6.0#orcuts[iii]
                #psi_nm = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                isize += Mcut
                iii += 1
        
    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    Av_coeffs = np.zeros(Tsize)
    Regr_Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                if l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
                    if response: Regr_Av_coeffs[i] = regr_av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1

    # add the average spherical coefficients to the predictions 
    if response:
        pred_coefs += Regr_Av_coeffs
    else:
        pred_coefs += Av_coeffs

    # save predicted coefficients
    np.save(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(iconf)+".npy",pred_coefs)

    # compute predicted density projections <phi|rho>
    pred_projs = np.dot(overl,pred_coefs)

    # compute error
    error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
    error_density += error
    ref_projs -= np.dot(overl,Av_coeffs)
    ref_coefs -= Av_coeffs
    var = np.dot(ref_coefs,ref_projs)
    variance += var
    print(iconf+1, ":", "error =", np.sqrt(error/var)*100, "% RMSE", flush=True)

    # UNCOMMENT TO CHECK PREDICTIONS OF <phi|rho-rho_0>
    # ------------------------------------------------- 
    #pred_projs = np.dot(overl,pred_coefs-Av_coeffs)
    #av_projs = np.dot(overl,Av_coeffs)
    #iaux = 0
    #for iat in range(natoms[iconf]):
    #    spe = atomic_symbols[iconf][iat]
    #    for l in range(lmax[spe]+1):
    #        for n in range(nmax[(spe,l)]):
    #            for im in range(2*l+1):
    #                if l==5 and im==0:
    #                    print(pred_projs[iaux],ref_projs[iaux])
    #                iaux += 1

if inp.parallel:
    error_density = comm.allreduce(error_density)
    variance = comm.allreduce(variance)
if (rank == 0): print("")
if (rank == 0): print("% RMSE =", 100*np.sqrt(error_density/variance))
