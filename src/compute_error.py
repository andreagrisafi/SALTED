import sys
import numpy as np
import scipy
from scipy import special
import math
import time
import ase
from ase import io
from ase.io import read
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-s",   "--testset",     type=int, default=1, help="test dataset selection")
    parser.add_argument("-f",   "--trainfrac"  , type=float, default=1.0, help="training set fraction")
    parser.add_argument("-r",   "--regular"  , type=float, default=1e-06, help="regularization")
    parser.add_argument("-m",   "--msize"  ,     type=int, default=100, help="number of reference environments")
    parser.add_argument("-rc",   "--cutoffradius"  , type=float, default=4.0, help="soap cutoff")
    parser.add_argument("-sg",   "--sigmasoap"  , type=float, default=0.3, help="soap sigma")
    parser.add_argument("-jit",   "--jitter"  , type=float, default=1e-10, help="jitter")
    args = parser.parse_args()
    return args

def set_variable_values_contraction(args):
    s = args.testset
    f = args.trainfrac
    r = args.regular
    m = args.msize
    rc = args.cutoffradius
    sg = args.sigmasoap
    jit = args.jitter
    return [s,f,r,m,rc,sg,jit]

args = add_command_line_arguments_contraction("predict density")
[nset,frac,reg,M,rc,sigma_soap,jit] = set_variable_values_contraction(args)

bohr2ang = 0.529177249
#========================== system definition
filename = "coords_1000.xyz"
xyzfile = read(filename,":")
ndata = len(xyzfile)
#======================= system parameters
coords = []
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int) 
for i in xrange(len(xyzfile)):
    coords.append(np.asarray(xyzfile[i].get_positions(),float)/bohr2ang)
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

#================== species dictionary
spe_dict = {}
spe_dict[0] = "H"
spe_dict[1] = "O"
#====================================== reference environments
fps_indexes = np.loadtxt("SELECTIONS/refs_selection_"+str(M)+".txt",int)
fps_species = np.loadtxt("SELECTIONS/spec_selection_"+str(M)+".txt",int)
#============== angular
lmax = {}
llmax = 3
lmax["O"] = 3
lmax["H"] = 2
nnmax = 10
nmax = {}
# oxygen
nmax[("O",0)] = 10
nmax[("O",1)] = 7
nmax[("O",2)] = 5
nmax[("O",3)] = 2
# hydrogen
nmax[("H",0)] = 4
nmax[("H",1)] = 3
nmax[("H",2)] = 2

# load predicted coefficients for test structures
trainrangetot = np.loadtxt("SELECTIONS/training_selection.txt",int)
testrange = np.setdiff1d(range(ndata),trainrangetot)
ntest = len(testrange)
natoms_test = natoms[testrange]

coeffs = np.load("PREDICTIONS/prediction_trainfrac"+str(frac)+"_M"+str(M)+"_reg"+str(reg)+"_jit"+str(jit)+".npy")

av_coefs = {}
for spe in ["H","O"]:
    av_coefs[spe] = np.load("AVERAGES/"+str(spe)+".npy")

itest=0
error_density = 0.0
variance = 0.0
for iconf in testrange:
    print "-------------------------------"  
    print "iconf = ", iconf 
    coordinates = coords[iconf]
    atoms = atomic_symbols[iconf]
    valences = atomic_valence[iconf]
    nele = np.sum(valences)
    #================================================
    projs_ref = np.load("PROJS_NPY/projections_conf"+str(iconf)+".npy")
    overl = np.load("OVER_NPY/overlap_conf"+str(iconf)+".npy")
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
                        coefficients[icoeff] = coeffs[itest,iat,l,n,im] + av_coefs[atoms[iat]][n]
                        averages[icoeff] = av_coefs[atoms[iat]][n]
                    else:
                        coefficients[icoeff] = coeffs[itest,iat,l,n,im] 
                    icoeff +=1
    projections = np.dot(overl,coefficients)
    #================================================
    error = np.dot(coefficients-coeffs_ref,projections-projs_ref)
    error_density += error 
    projs_ref -= np.dot(overl,averages)
    coeffs_ref -= averages
    var = np.dot(coeffs_ref,projs_ref)
    variance += var 
    print "error =", np.sqrt(error/var)*100, "%"
    itest+=1

print ""
print "% RMSE = ", 100*np.sqrt(error_density/variance)
