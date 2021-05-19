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
testrange = np.array(range(ndata),int)
ntest = len(testrange)
natoms_test = natoms[testrange]

coeffs = np.load("pred_coeffs.npy")

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load(inp.path2ref+"averages_"+str(spe)+".npy")

dirpath = os.path.join(inp.path2data, "predictions")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

itest=0
error_density = 0.0
variance = 0.0
f = open("errors_predictions.dat","w")
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
    #================================================
    coefficients = np.zeros(size_coeffs,float)
    averages = np.zeros(size_coeffs,float)
    icoeff = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            for n in xrange(nmax[(atoms[iat],l)]):
                for im in xrange(2*l+1):
                    if l==0:
                        coefficients[icoeff] = coeffs[itest,iat,l,n,im] #+ av_coefs[atoms[iat]][n]
                        averages[icoeff] = av_coefs[atoms[iat]][n]
                    else:
                        coefficients[icoeff] = coeffs[itest,iat,l,n,im] 
                    icoeff +=1
    projections = np.dot(overl,coefficients)
    np.save(inp.path2data+"predictions/prediction_conf"+str(iconf)+".npy",projections)
    #================================================
    error = np.dot(coefficients-coeffs_ref,projections-projs_ref)
    error_density += error 
    projs_ref -= np.dot(overl,averages)
    coeffs_ref -= averages
    var = np.dot(coeffs_ref,projs_ref)
    variance += var 
    print >> f, iconf+1, ":", np.sqrt(error/var)*100, "% RMSE"
    itest+=1

f.close()

print "Error =", np.sqrt(error_density/ntest)
print "STD =", np.sqrt(variance/ntest)
print "% RMSE =", 100*np.sqrt(error_density/variance)
