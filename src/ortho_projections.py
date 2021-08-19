import os
import sys
import numpy as np
import scipy
from scipy import special
import time
import ase
import argparse
from ase import io
from ase.io import read
from scipy.sparse.linalg import eigsh


def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-j1", "--istart", type=int, default=0, help="starting index")
    parser.add_argument("-j2", "--iend",   type=int, default=0, help="ending index")
    parser.add_argument("-iconf", "--iselection",   type=int, default=0, help="selected conf")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("dataset subselection")
# dataset slice boundaries 
istart = args.istart-1
iend = args.iend
isel = args.iselection # 0 based

import basis
sys.path.insert(0, './')
import inp

ocut = inp.overcut

# read species
spelist = inp.species

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
for spe in spelist:
    llist.append(lmax[spe])
llmax = max(llist)

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

#======================= system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int) 
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))

#==================== species array
nspecies = len(spelist)

natoms_spe = {}
for iconf in xrange(ndata):
    for spe in spelist:
        natoms_spe[(iconf,spe)] = 0

for iconf in xrange(ndata):
    species = atomic_symbols[iconf]
    for iat in xrange(natoms[iconf]):
        spe = species[iat]
        natoms_spe[(iconf,spe)] += 1

nenv = {}
for spe in spelist:
    nenv[spe] = 0
    for iconf in xrange(ndata):
        nenv[spe] += natoms_spe[iconf,spe]

print "computing averages..."
# init averages
av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)

# compute averages
for iconf in xrange(ndata):
    species = atomic_symbols[iconf]
    #==================================================
    Coef = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
#    Proj = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
#    Over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
#    Coef = np.linalg.solve(Over,Proj)
#    np.save(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy",Coef)
    #==================================================
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = species[iat] 
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       av_coefs[spe][n] += Coef[i]
                    i += 1


# save averages
for spe in spelist:
    av_coefs[spe] /= nenv[spe]
    np.save("averages_"+str(spe)+".npy",av_coefs[spe])
    for l in xrange(lmax[spe]+1):
        for n in xrange(nmax[(spe,l)]):
            dirpath = os.path.join(inp.path2qm+"projections", "spe"+str(spe)+"_l"+str(l)+"_n"+str(n))
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)

print "computing orthogonal projections..."
#for iconf in xrange(istart,iend):
#for iconf in [isel]:
for iconf in xrange(ndata):
    print iconf+1
    
    start = time.time()
    species = atomic_symbols[iconf]
    # init orthogonal projections
    projs = {}
    for spe in spelist:
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                projs[(spe,l,n)] = np.zeros((natoms_spe[iconf,spe],(2*l+1)))
    # compute coefficients
    Coef = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")
    #Proj = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
    Over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    #Coef = np.linalg.solve(Over,Proj)
    # remove L=0 average
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = species[iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       Coef[i] -= av_coefs[spe][n]
                    i += 1
    # compute baselined projections
    DProj = np.dot(Over,Coef)
    # compute orthogonalization matrix
    eigenvalues, unitary = np.linalg.eigh(Over)
    eigenvalues = eigenvalues[eigenvalues>ocut]
    Mcut = len(eigenvalues)
    sqrteigen = np.sqrt(eigenvalues) 
    diagoverlap = np.diag(1.0/sqrteigen)
    orthomatrix = np.dot(np.conj(unitary[:,-Mcut:]),np.dot(diagoverlap,unitary[:,-Mcut:].T))
    np.save(inp.path2qm+"overlaps/orthomatrix_"+str(iconf)+".npy",orthomatrix)
    # orthogonalize projections
    OProj = np.dot(orthomatrix,DProj)
    # init species counting 
    specount = {}
    for spe in spelist:
        specount[spe] = 0
    # fill array of orthogonal projections
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = species[iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    projs[(spe,l,n)][specount[spe],im] = OProj[i] 
                    i += 1
        specount[spe] += 1
    # save orthogonal projections
    for spe in spelist:
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                np.save(inp.path2qm+"projections/spe"+str(spe)+"_l"+str(l)+"_n"+str(n)+"/ortho_projections_conf"+str(iconf)+".npy",projs[(spe,l,n)].reshape(natoms_spe[(iconf,spe)]*(2*l+1)))

    print (time.time()-start)/60.0, "minutes"
