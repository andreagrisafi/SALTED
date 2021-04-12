import os
import sys
import time
import numpy as np
from ase.io import read

import basis

sys.path.insert(0, './')
import inp

# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i]

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# number of sparse environments
M = inp.Menv

#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int) 
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

# load predicted coefficients for test structures
trainrangetot = np.loadtxt("training_set.txt",int)
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]

ortho_coeffs = np.load("predictions.npy")

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

dirpath = os.path.join(inp.path2data, "predictions")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

itest=0
Oerror_density = 0.0
variance = 0.0
print "Estimating prediction error ..."
for iconf in testrange:
    atoms = atomic_symbols[iconf]
    valences = atomic_valence[iconf]
    nele = np.sum(valences)
    #================================================
    projs_ref = np.load(inp.path2data+"projections/projections_conf"+str(iconf)+".npy")
    overl = np.load(inp.path2data+"overlaps/overlap_conf"+str(iconf)+".npy")
    coeffs_ref = np.linalg.solve(overl,projs_ref)
    size_coeffs = coeffs_ref.shape
    # compute orthogonalization matrix
    eigenvalues, unitary = np.linalg.eig(overl)
    sqrteigen = np.sqrt(eigenvalues)
    diagoverlap = np.diag(1.0/sqrteigen)
    orthomatrix = np.dot(np.conj(unitary),np.dot(diagoverlap,unitary.T))
    #newoverlap = np.dot(np.conj(unitary),np.dot(diagoverlap,unitary.T))
    #orthomatrix = np.linalg.inv(newoverlap)
    OCoeffs = np.zeros(len(overl))
    i = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            for n in xrange(nmax[(atoms[iat],l)]):
                for im in xrange(2*l+1):
                    OCoeffs[i] = ortho_coeffs[itest,iat,l,n,im]
                    i+=1
    OCoef = np.dot(orthomatrix,OCoeffs)
    #================================================
    coefficients = np.zeros(size_coeffs,float)
    averages = np.zeros(size_coeffs,float)
    icoeff = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            for n in xrange(nmax[(atoms[iat],l)]):
                for im in xrange(2*l+1):
                    if l==0:
                        OCoef[icoeff] += av_coefs[atoms[iat]][n]
                        averages[icoeff] = av_coefs[atoms[iat]][n]
                    icoeff +=1
    OProj = np.dot(overl,OCoef)
    #================================================
    Oerror = np.dot(OCoef-coeffs_ref,OProj-projs_ref)
    Oerror_density += Oerror 
    projs_ref -= np.dot(overl,averages)
    coeffs_ref -= averages
    var = np.dot(coeffs_ref,projs_ref)
    variance += var 
    itest+=1

print "% RMSE =", 100*np.sqrt(Oerror_density/variance)
