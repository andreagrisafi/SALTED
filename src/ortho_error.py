import os
import time
import numpy as np
import sys
sys.path.insert(0, './')
import inp
from sys_utils import read_system

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut

pdir = inp.valcdir

# load predicted coefficients for test structures
trainrangetot = np.loadtxt("training_set.txt",int)
ntrain = int(inp.trainfrac*len(trainrangetot))
testrange = np.setdiff1d(list(range(ndata)),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]

dirpath = os.path.join(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N_"+str(ntrain))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

ortho_coeffs = np.load(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/ortho-predictions_N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+".npy")

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

itest=0
Oerror_density = 0.0
variance = 0.0
preds = np.zeros((ntest,natmax,llmax+1,nnmax,2*llmax+1))
for iconf in testrange:
#    print(iconf+1)
    start = time.time()
    atoms = atomic_symbols[iconf]
    #================================================
    projs_ref = np.load(inp.path2qm+inp.projdir+"projections_conf"+str(iconf)+".npy")
    coeffs_ref = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    size_coeffs = coeffs_ref.shape
    # compute orthogonalization matrix
    orthomatrix = np.load(inp.path2qm+inp.ovlpdir+"orthomatrix_"+str(iconf)+".npy")
    OCoeffs = np.zeros(size_coeffs)
    i = 0
    for iat in range(natoms[iconf]):
        for l in range(lmax[atoms[iat]]+1):
            for n in range(nmax[(atoms[iat],l)]):
                for im in range(2*l+1):
                    OCoeffs[i] = ortho_coeffs[itest,iat,l,n,im]
                    i+=1
    OCoef = np.dot(orthomatrix,OCoeffs)
    #================================================
    coefficients = np.zeros(size_coeffs,float)
    averages = np.zeros(size_coeffs,float)
    icoeff = 0
    for iat in range(natoms[iconf]):
        for l in range(lmax[atoms[iat]]+1):
            for n in range(nmax[(atoms[iat],l)]):
                for im in range(2*l+1):
                    if l==0:
                        OCoef[icoeff] += av_coefs[atoms[iat]][n]
                        averages[icoeff] = av_coefs[atoms[iat]][n]
                    preds[itest,iat,l,n,im] = OCoef[icoeff]
                    icoeff +=1
    np.save(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(iconf)+".npy",OCoef)
    overl = np.load(inp.path2qm+inp.ovlpdir+"overlap_conf"+str(iconf)+".npy")
    OProj = np.dot(overl,OCoef)
    #================================================
    Oerror = np.dot(OCoef-coeffs_ref,OProj-projs_ref)
    Oerror_density += Oerror 
    projs_ref -= np.dot(overl,averages)
    coeffs_ref -= averages
    var = np.dot(coeffs_ref,projs_ref)
    variance += var
    print(iconf+1, ":", np.sqrt(Oerror/var)*100, "% RMSE",flush=True)
#    print("time:",time.time()-start)
    itest+=1


print("% RMSE =", 100*np.sqrt(Oerror_density/variance))
#np.save(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/pred-coeffs_N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+".npy",preds)
